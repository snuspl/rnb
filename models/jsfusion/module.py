"""A PyTorch implementation of the JSFusion model.
See https://github.com/yj-yu/lsmdc for the original
TensorFlow implementation from the authors.
A Joint Sequence Fusion Model for Video Question Answering and Retrieval,
Yu et al., ECCV 2018.
"""
import torch
import torch.nn.functional as F
import numpy as np

import time
import os

from models.jsfusion.attention import add_timing_signal_nd
import hickle as hkl
from torchvision import models
import math

class ResNetFeatureExtractor(torch.nn.Module):
  def __init__(self, num_frames = 40):
    super(ResNetFeatureExtractor, self).__init__()
    self.resnet = models.resnet152(pretrained=True)
    self.num_frames = num_frames

    module_list = list(self.resnet.children())
    self.pool = torch.nn.Sequential(*module_list[:-1])
    self.in_features = module_list[-1].in_features # 2048 for ResNet152

  def forward(self, tensors):
    (frames,), filename = tensors
    resnet_output = self.pool(frames)
    resnet_output = resnet_output.view(resnet_output.shape[0], resnet_output.shape[1])
    # TODO handle the case when resnet_output.shape[0] < num_frames (fill zeros)
    resnet_output = resnet_output[:self.num_frames, :]

    return ((resnet_output,), filename)


class MCModel(torch.nn.Module):

  def __init__(self, device, dropout_prob = 0.5, video_channels = 2048, num_frames = 40):
    super(MCModel, self).__init__()

    self.device = device

    self.num_frames = num_frames
    self.register_buffer('mask', torch.ones((self.num_frames), dtype=torch.float32))
    self.register_buffer('one', torch.tensor(1, dtype=torch.int32))
    self.register_buffer('signal', add_timing_signal_nd(self.num_frames, video_channels))

    self.dropout = torch.nn.Dropout(p=dropout_prob)
    self.conv1 = torch.nn.Conv2d(2048, 2048, [3, 1], padding=(1, 0))
    self.relu1 = torch.nn.ReLU()
    self.bn1 = torch.nn.BatchNorm2d(2048, eps=0.001, momentum=0.001)
    self.conv2 = torch.nn.Conv2d(2048, 2048, [3, 1], padding=(1, 0))
    self.relu2 = torch.nn.ReLU()
    self.bn2 = torch.nn.BatchNorm2d(2048, eps=0.001, momentum=0.001)
    self.conv3 = torch.nn.Conv2d(2048, 2048, [3, 1], padding=(1, 0))
    self.relu3 = torch.nn.ReLU()
    self.bn3 = torch.nn.BatchNorm2d(2048, eps=0.001, momentum=0.001)

    self.sigmoid = torch.nn.Sigmoid()

    self.fc4 = torch.nn.Linear(1024+video_channels, 512)
    self.tanh4 = torch.nn.Tanh()
    self.bn4 = torch.nn.BatchNorm1d(512, eps=0.001, momentum=0.001)

    embedding_matrix = hkl.load(os.path.join(os.environ['LSMDC_PATH'], 'hkls/common_word_matrix_py3.hkl'))
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    self.register_buffer('embedding_matrix', embedding_matrix)


    self.lstm = torch.nn.LSTM(300, 512, 1, batch_first=True, dropout=dropout_prob, bidirectional=True)
    self.fc5 = torch.nn.Linear(512*2, 512)
    self.tanh5 = torch.nn.Tanh()
    self.bn5 = torch.nn.BatchNorm1d(512, eps=0.001, momentum=0.001)


    self.fusion_fc1 = torch.nn.Linear(512, 512)
    self.fusion_tanh1 = torch.nn.Tanh()
    self.fusion_bn1 = torch.nn.BatchNorm2d(512, eps=0.001, momentum=0.001)
    self.fusion_gate2 = torch.nn.Linear(512, 1)
    self.fusion_sigmoid2 = torch.nn.Sigmoid()
    self.fusion_bn2 = torch.nn.BatchNorm2d(1, eps=0.001, momentum=0.001)

    self.fusion_fc3 = torch.nn.Linear(512, 512)
    self.fusion_tanh3 = torch.nn.Tanh()
    self.fusion_bn3 = torch.nn.BatchNorm2d(512, eps=0.001, momentum=0.001)
    self.fusion_fc4 = torch.nn.Linear(512, 512)
    self.fusion_tanh4 = torch.nn.Tanh()
    self.fusion_bn4 = torch.nn.BatchNorm2d(512, eps=0.001, momentum=0.001)


    self.fusion_next_conv1 = torch.nn.Conv2d(512, 256, [3, 3])
    self.fusion_next_tanh1 = torch.nn.Tanh()
    self.fusion_next_convalp1 = torch.nn.Conv2d(512, 1, [3, 3])
    self.fusion_next_tanhalp1 = torch.nn.Tanh()
    self.fusion_next_gate2 = torch.nn.Sigmoid()

    self.fusion_next_conv2 = torch.nn.Conv2d(256, 256, [3, 3])
    self.fusion_next_tanh2 = torch.nn.Tanh()
    self.fusion_next_convalp2 = torch.nn.Conv2d(256, 1, [3, 3])
    self.fusion_next_tanhalp2 = torch.nn.Tanh()
    self.fusion_next_gate3 = torch.nn.Sigmoid()

    self.fusion_next_conv3 = torch.nn.Conv2d(256, 256, [3, 3], stride=(2, 2))
    self.fusion_next_tanh3 = torch.nn.Tanh()
    self.fusion_next_convalp3 = torch.nn.Conv2d(256, 1, [3, 3], stride=(2, 2))
    self.fusion_next_tanhalp3 = torch.nn.Tanh()
    self.fusion_next_gate4 = torch.nn.Sigmoid()


    self.final_fc1 = torch.nn.Linear(256, 256)
    self.final_tanh1 = torch.nn.Tanh()
    self.final_bn1 = torch.nn.BatchNorm1d(256, eps=0.001, momentum=0.001)
    self.final_fc2 = torch.nn.Linear(256, 256)
    self.final_tanh2 = torch.nn.Tanh()
    self.final_bn2 = torch.nn.BatchNorm1d(256, eps=0.001, momentum=0.001)
    self.final_fc3 = torch.nn.Linear(256, 128)
    self.final_tanh3 = torch.nn.Tanh()
    self.final_bn3 = torch.nn.BatchNorm1d(128, eps=0.001, momentum=0.001)
    self.final_fc4 = torch.nn.Linear(128, 1)
    self.final_bn4 = torch.nn.BatchNorm1d(1, eps=0.001, momentum=0.001)
    self.word2idx = hkl.load(os.path.join(os.environ['LSMDC_PATH'], 'hkls/common_word_to_index_py3.hkl'))
   

  def video_embeddings(self, video, mask):
    # BxLxC
    embedded_feat_tmp = video + self.signal
    embedded_feat = embedded_feat_tmp * torch.unsqueeze(mask, 2)
    embedded_feat_drop = self.dropout(embedded_feat)

    # BxCxL
    video_emb = embedded_feat_drop.permute(0, 2, 1)

    # BxCxLx1
    video_emb_unsqueezed = torch.unsqueeze(video_emb, 3)

    conv1 = self.conv1(video_emb_unsqueezed)
    relu1 = self.relu1(conv1)
    bn1 = self.bn1(relu1)

    conv2 = self.conv2(bn1)
    relu2 = self.relu2(conv2)
    bn2 = self.bn2(relu2)

    conv3 = self.conv3(bn2)
    relu3 = self.relu3(conv3)
    bn3 = self.bn3(relu3)

    # Bx2048xL
    outputs = torch.squeeze(bn3, 3)
    input_pass = outputs[:, 0:1024, :]
    input_gate = outputs[:, 1024:, :]
    input_gate = self.sigmoid(input_gate)
    outputs = input_pass * input_gate

    # Bx(C+2048)xL
    outputs = torch.cat([outputs, video_emb], dim=1)

    # BxLx(C+2048)
    outputs = outputs.permute(0, 2, 1)

    # BxLx512
    fc4 = self.fc4(outputs)
    tanh4 = self.tanh4(fc4)

    # Bx512xL
    tanh4 = tanh4.permute(0, 2, 1)
    bn4 = self.bn4(tanh4)

    masked_outputs = bn4 * torch.unsqueeze(mask, 1)
    return masked_outputs


  def word_embeddings(self, captions, caption_masks):
    # 5BxL
    captions = captions.view(-1, captions.shape[-1])
    # 5BxLxH
    seq_embeddings = self.embeddings(captions)

    # 5BxLx1
    caption_masks = caption_masks.view(-1, caption_masks.shape[-1], 1)

    # 5BxLxH
    embedded_sentence = seq_embeddings * caption_masks

    # 5BxLx1024
    outputs, _ = self.lstm(embedded_sentence)

    # 5BxLx512
    fc5 = self.fc5(outputs)
    tanh5 = self.tanh5(fc5)

    # 5Bx512xL
    tanh5 = tanh5.permute(0, 2, 1)
    bn5 = self.bn5(tanh5)

    rnn_output = bn5 * caption_masks.view(caption_masks.shape[0], 1, caption_masks.shape[1])
    return rnn_output


  def fusion(self, v, w, mask, caption_masks):
    # 5Bx512xL
    v = v.repeat(5, 1, 1)

    # 5Bx512xLx1
    vv = torch.unsqueeze(v, 3)

    # 5Bx512x1xL
    ww = torch.unsqueeze(w, 2)

    # 5Bx512xLxL
    cnn_repr = vv * ww

    # 5BxLxLx512
    cnn_repr = cnn_repr.permute(0, 2, 3, 1) 

    # 5BxLxLx512
    fc1 = self.fusion_fc1(cnn_repr)
    tanh1 = self.fusion_tanh1(fc1)
    tanh1 = tanh1.permute(0, 3, 1, 2)
    bn1 = self.fusion_bn1(tanh1)
    bn1 = bn1.permute(0, 2, 3, 1)

    # 5BxLxLx1
    gate2 = self.fusion_gate2(bn1)
    sigmoid2 = self.fusion_sigmoid2(gate2)
    sigmoid2 = sigmoid2.permute(0, 3, 1, 2)
    bn2 = self.fusion_bn2(sigmoid2)
    # 5Bx1xLxL

    # 5BxLxLx512
    fc3 = self.fusion_fc3(cnn_repr)
    tanh3 = self.fusion_tanh3(fc3)
    tanh3 = tanh3.permute(0, 3, 1, 2)
    bn3 = self.fusion_bn3(tanh3)
    bn3 = bn3.permute(0, 2, 3, 1)

    # 5BxLxLx512
    fc4 = self.fusion_fc4(bn3)
    tanh4 = self.fusion_tanh4(fc4)
    tanh4 = tanh4.permute(0, 3, 1, 2)
    bn4 = self.fusion_bn4(tanh4)
    # 5Bx512xLxL

    # 5Bx512xLxL
    output1 = bn4 * bn2

    # Bx1xLx1
    shape = mask.shape
    mask = torch.reshape(mask, (shape[0], 1, shape[1], 1))

    # 5Bx1xLx1
    mask = mask.repeat(5, 1, 1, 1)

    # 5Bx1x1xL
    shape = caption_masks.shape
    caption_masks = torch.reshape(caption_masks, (shape[0] * shape[1], 1, 1, shape[2]))

    # 5Bx512xLxL
    output1 = output1 * mask * caption_masks

    return output1


  def fusion_next(self, output1, mask, caption_masks):
    # 5BxL
    caption_masks = torch.reshape(caption_masks, (-1, caption_masks.shape[-1]))

    cut_mask_list = []
    cut_caption_masks_list = []
    cut_mask = mask[:, :-2]
    cut_mask[:, -1] = 1.

    # Bx(L-2)
    cut_mask_list.append(cut_mask.repeat(5, 1))
    cut_caption_masks = caption_masks[:, 2:]
    cut_caption_masks[:, 0] = 1.
    # 5Bx(L-2)
    cut_caption_masks_list.append(cut_caption_masks)

    cut_mask = cut_mask[:, :-2]
    cut_mask[:, -1] = 1.
    # Bx(L-4)
    cut_mask_list.append(cut_mask.repeat(5, 1))
    cut_caption_masks = cut_caption_masks[:, 2:]
    cut_caption_masks[:, 0] = 1.
    # 5Bx(L-4)
    cut_caption_masks_list.append(cut_caption_masks)


    max_len = (mask.shape[1] - 5) // 2
    cut_mask_len = (torch.sum(cut_mask, 1, dtype=torch.int32) - 1) / 2
    cut_mask_len = torch.max(cut_mask_len, self.one)
    cut_caption_masks_len = (torch.sum(cut_caption_masks, 1, dtype=torch.int32) - 1) / 2
    cut_caption_masks_len = torch.max(cut_caption_masks_len, self.one)


    cut_mask_indices = [i for i in range(cut_mask.shape[1]) if i % 2 == 1 and i < cut_mask.shape[1] - 1]
    cut_mask_indices = torch.tensor(cut_mask_indices)
    cut_mask_indices = cut_mask_indices.to(device=self.device, non_blocking=True)

    # cut_mask = torch.tensor([([0]*(max_len - l) + [1]*l) for l in cut_mask_len.cpu().numpy()], dtype=torch.float32)
    cut_mask = torch.index_select(cut_mask, 1, cut_mask_indices)

    cut_caption_masks_indices = [i for i in range(cut_caption_masks.shape[1]) if i % 2 == 1 and i > 1]
    cut_caption_masks_indices = torch.tensor(cut_caption_masks_indices)
    cut_caption_masks_indices = cut_caption_masks_indices.to(device=self.device, non_blocking=True)


    # cut_caption_masks = torch.tensor([([1]*l + [0]*(max_len - l)) for l in cut_caption_masks_len.cpu().numpy()], dtype=torch.float32)
    cut_caption_masks = torch.index_select(cut_caption_masks, 1, cut_caption_masks_indices)

    cut_mask_list.append(cut_mask.repeat(5, 1))
    cut_caption_masks_list.append(cut_caption_masks)


    # 5Bx256x(L-2)x(L-2)
    conv1 = self.fusion_next_conv1(output1)
    tanh1 = self.fusion_next_tanh1(conv1)

    # 5Bx1x(L-2)x(L-2)
    convalp1 = self.fusion_next_convalp1(output1)
    tanhalp1 = self.fusion_next_tanhalp1(convalp1)
    gate2 = self.fusion_next_gate2(tanhalp1)

    # 5Bx256x(L-2)x(L-2)
    output2 = tanh1 * gate2

    # (5B, L-2)
    shape = cut_mask_list[0].shape
    mask = torch.reshape(cut_mask_list[0], (shape[0], 1, shape[1], 1))
    # (5B, L-2)
    shape = cut_caption_masks_list[0].shape
    caption_masks = torch.reshape(cut_caption_masks_list[0], (shape[0], 1, 1, shape[1]))

    # 5Bx256x(L-2)x(L-2)
    output2 = output2 * mask * caption_masks


    # 5Bx256x(L-4)x(L-4)
    conv2 = self.fusion_next_conv2(output2)
    tanh2 = self.fusion_next_tanh2(conv2)

    # 5Bx1x(L-4)x(L-4)
    convalp2 = self.fusion_next_convalp2(output2)
    tanhalp2 = self.fusion_next_tanhalp2(convalp2)
    gate3 = self.fusion_next_gate3(tanhalp2)

    # 5Bx256x(L-4)x(L-4)
    output3 = tanh2 * gate3

    # (5B, L-4)
    shape = cut_mask_list[1].shape
    mask = torch.reshape(cut_mask_list[1], (shape[0], 1, shape[1], 1))
    # (5B, L-4)
    shape = cut_caption_masks_list[1].shape
    caption_masks = torch.reshape(cut_caption_masks_list[1], (shape[0], 1, 1, shape[1]))

    # 5Bx256x(L-4)x(L-4)
    output3 = output3 * mask * caption_masks


    # 5Bx256xhalfxhalf
    conv3 = self.fusion_next_conv3(output3)
    tanh3 = self.fusion_next_tanh3(conv3)

    # 5Bx1xhalfxhalf
    convalp3 = self.fusion_next_convalp3(output3)
    tanhalp3 = self.fusion_next_tanhalp3(convalp3)
    gate4 = self.fusion_next_gate4(tanhalp3)

    # 5Bx256xhalfxhalf
    output4 = tanh3 * gate4

    # (5B, half)
    shape = cut_mask_list[2].shape
    mask = torch.reshape(cut_mask_list[2], (shape[0], 1, shape[1], 1))
    # (5B, half)
    shape = cut_caption_masks_list[2].shape
    caption_masks = torch.reshape(cut_caption_masks_list[2], (shape[0], 1, 1, shape[1]))

    # 5Bx256xhalfxhalf
    output4 = output4 * mask * caption_masks


    # 5B
    valid = torch.sum(cut_mask_list[2], 1) * torch.sum(cut_caption_masks_list[2], 1)
    sum_state = torch.sum(output4, (2, 3)) / torch.unsqueeze(valid, 1)

    return sum_state


  def final(self, fusion_next):
    # 5Bx256
    a = self.final_fc1(fusion_next)
    a = self.final_tanh1(a)
    a = self.final_bn1(a)

    # 5Bx256
    a = self.final_fc2(a)
    a = self.final_tanh2(a)
    a = self.final_bn2(a)

    # 5Bx128
    a = self.final_fc3(a)
    a = self.final_tanh3(a)
    a = self.final_bn3(a)

    # 5Bx1
    a = self.final_fc4(a)
    a = self.final_bn4(a)

    return torch.reshape(-a, (-1, 5))
  
  
  def parse_sentences(self, word2idx, mc, max_length):
    import numpy as np
    def sentence_to_words(sentence):
      from models.jsfusion.data_util import clean_str
      try:
        words = clean_str(sentence).split()
      except:
        print('[ERROR] sentence is broken: ' + sentence)
        sys.exit(1)
      
      for w in words:
        if not w:
          continue
        yield w
      
    def sentence_to_matrix(word2idx, sentence, max_length):
      indices = [word2idx[w] for w in
                 sentence_to_words(sentence)
                 if w in word2idx]
      length = min(len(indices), max_length)
      return indices[:length]

    with open(mc, 'r') as f:
      sentences = [sentence_to_matrix(word2idx, f.readline().strip(), max_length)
                   for _ in range(5)]
  
    sentences_tmps = []
    for sent in sentences:
      sentences_tmps.append(sent + [0] * (max_length - len(sent)))
    sentences = sentences_tmps
  
    sentence_masks = []
    for sent in sentences:
      sentence_masks.append([(1 if s != 0 else 0) for s in sent])
  
    sentences = np.asarray(sentences, dtype=np.int32)
    sentences = np.reshape(sentences, (1, 5, -1))
  
    sentence_masks = np.asarray(sentence_masks, dtype=np.float32)
    sentence_masks = np.reshape(sentence_masks, (1, 5, -1))

    return sentences, sentence_masks


  def forward(self, tensors):
    """Main inference function for the model.
    resnet_output: torch.Tensor device='cuda' shape=(BxLx2048) dtype=float32
    """
    (resnet_output,), filename = tensors 

    self.mask = self.mask.squeeze(0)
    if resnet_output.shape[0] < self.num_frames:
      more_zeros = self.num_frames - resnet_output.shape[0]
      self.mask[:more_zeros] = 0.

    elif resnet_output.shape[0] > self.num_frames:
      print('Movie %s is over %d frames' % (video, self.num_frames))
      self.mask = self.mask[:self.num_frames]

    # mask: torch.Tensor shape=(BxL) dtype=float32
    self.mask = torch.unsqueeze(self.mask, 0)

    mc_path = os.path.join(os.environ['LSMDC_PATH'], 'texts', os.path.splitext(filename)[0] + '.txt')

    # sentences: np.ndarray shape=(Bx5xL) dtype=int32
    # sentence_masks: np.ndarray shape=(Bx5xL) dtype=float32
    sentences, sentence_masks = self.parse_sentences(self.word2idx, mc_path, self.num_frames)
    sentences = torch.tensor(sentences, dtype=torch.long).to(self.device)
    sentence_masks = torch.tensor(sentence_masks, dtype=torch.float32).to(self.device)

    # Bx512xL
    d1v = self.video_embeddings(resnet_output, self.mask)

    # 5Bx512xL
    self.embeddings = torch.nn.Embedding.from_pretrained(self.embedding_matrix, freeze=False)
    d1w = self.word_embeddings(sentences, sentence_masks)

    # 5Bx512xLxL
    fusion = self.fusion(d1v, d1w, self.mask, sentence_masks)

    # 5Bx256
    fusion_next = self.fusion_next(fusion, self.mask, sentence_masks)

    # Bx5
    logits = self.final(fusion_next)

    # B
    winners = torch.argmax(logits, dim=1)

    return ((winners,), None)
