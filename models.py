import torch
import torch.nn as nn
import numpy as np


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # 输出层
        self.fc_out = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # LSTM 输出
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_size]

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 输出层
        output = self.fc_out(last_output)

        return output


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size

        # 双向 LSTM 参数
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

        # 输出层
        self.fc_out = nn.Linear(2 * hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        batch_size = x.size(0)

        # LSTM 输出
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, 2 * hidden_size]

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 输出层
        output = self.fc_out(last_output)

        return output


class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMWithAttention, self).__init__()

        self.hidden_size = hidden_size

        # 双向 LSTM 参数
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

        # Attention 层
        self.attention = nn.Linear(2 * hidden_size, 1)

        # 输出层
        self.fc_out = nn.Linear(2 * hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        batch_size = x.size(0)

        # LSTM 输出
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, 2 * hidden_size]

        # Attention 权重
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # 加权求和得到上下文向量
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # 输出层
        output = self.fc_out(context_vector)

        return output


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size

        # 使用整个 GRU 层而不是 GRUCell
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

        # 输出层
        self.fc_out = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # GRU 输出
        gru_out, _ = self.gru(x)  # gru_out: [batch_size, seq_len, hidden_size]

        # 取最后一个时间步的输出
        last_output = gru_out[:, -1, :]

        # 输出层
        output = self.fc_out(last_output)

        return output


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiGRU, self).__init__()

        self.hidden_size = hidden_size

        # 双向 GRU 参数
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)

        # 输出层
        self.fc_out = nn.Linear(2 * hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        batch_size = x.size(0)

        # GRU 输出
        gru_out, _ = self.gru(x)  # gru_out: [batch_size, seq_len, 2 * hidden_size]

        # 取最后一个时间步的输出
        last_output = gru_out[:, -1, :]

        # 输出层
        output = self.fc_out(last_output)

        return output


class BiGRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiGRUWithAttention, self).__init__()

        self.hidden_size = hidden_size

        # 双向 GRU 参数
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)

        # 注意力层
        self.attention = nn.Linear(2 * hidden_size, 1)

        # 输出层
        self.fc_out = nn.Linear(2 * hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        batch_size = x.size(0)

        # GRU 输出
        gru_out, _ = self.gru(x)  # gru_out: [batch_size, seq_len, 2 * hidden_size]

        # 注意力权重
        attention_weights = torch.softmax(self.attention(gru_out), dim=1)

        # 加权求和得到上下文向量
        context_vector = torch.sum(attention_weights * gru_out, dim=1)

        # 输出层
        output = self.fc_out(context_vector)

        return output


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, hidden_dim)

        # 位置编码
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 任务特定头
        self.death_head = nn.Linear(hidden_dim, 1)
        self.icu_los_class_head = nn.Linear(hidden_dim, 9)
        self.icu_los_reg_head = nn.Linear(hidden_dim, 1)
        self.hosp_los_class_head = nn.Linear(hidden_dim, 9)
        self.hosp_los_reg_head = nn.Linear(hidden_dim, 1)
        self.sofa_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        # 输入嵌入
        x = self.input_embedding(x)

        # 添加位置编码
        x = self.positional_encoding(x)

        # Transformer编码
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)

        # 全局平均池化
        x = x.mean(dim=1)

        # 任务输出
        death_prob = torch.sigmoid(self.death_head(x))
        icu_los_class = self.icu_los_class_head(x)
        icu_los_reg = self.icu_los_reg_head(x)
        hosp_los_class = self.hosp_los_class_head(x)
        hosp_los_reg = self.hosp_los_reg_head(x)
        sofa_score = self.sofa_head(x)

        return {
            'death_prob': death_prob,
            'icu_los_class': icu_los_class,
            'icu_los_reg': icu_los_reg,
            'hosp_los_class': hosp_los_class,
            'hosp_los_reg': hosp_los_reg,
            'sofa_score': sofa_score
        }


# 位置编码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)