import torch.nn as nn
from transformers import BertPreTrainedModel
import torch
from transformers import BertModel


class Bert(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(Bert, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pre_linear = nn.Linear(config.hidden_size, 300)
        self.activation = nn.SELU()
        self.reduce_fuse_linear = nn.Linear(600, 300)
        self.cos = nn.CosineSimilarity()

        self.rank_margin = nn.MarginRankingLoss(margin=0.4)

        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier_direct = nn.Linear(600, 1)

        self.init_weights()

    def forward(self, topic_id, tweet_text, labels=None):
        _, topic_id = self.bert(topic_id)
        topic_id = self.activation(topic_id)
        topic_id = self.pre_linear(topic_id)

        _, tweet_text = self.bert(tweet_text)
        tweet_text = self.activation(tweet_text)
        tweet_text = self.pre_linear(tweet_text)

        fused_tweet = torch.cat((topic_id + tweet_text, topic_id * tweet_text), dim=1)
        fused_tweet = self.reduce_fuse_linear(fused_tweet)

        cos_topic_id = self.cos(fused_tweet, topic_id)
        cos_tweet_text = self.cos(fused_tweet, tweet_text)

        outputs = (cos_topic_id, cos_tweet_text), None, None

        if labels is not None:
            labels[labels == 0] = -1
            loss_rank = self.rank_margin(cos_topic_id, cos_tweet_text, labels)

            outputs = (loss_rank,) + outputs
        return outputs
