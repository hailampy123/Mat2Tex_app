import segment
from PIL import Image, ImageOps
import numpy as np
import json
import re
import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

# Classified packages
import torchvision.transforms as transforms
from torchvision.models import resnet18



class Text(ABC):
    def __init__(self):
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2

    @abstractmethod
    def tokenize(self, formula: str):
        pass

    def int2text(self, x: Tensor):
        return " ".join([self.id2word[i] for i in x if i > self.eos_id])

    def text2int(self, formula: str):
        return torch.LongTensor([self.word2id[i] for i in self.tokenize(formula)])

class Text100k(Text):
    def __init__(self):
        super().__init__()
        self.id2word = json.load(open("100k_vocab.json", "r"))
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))
        self.TOKENIZE_PATTERN = re.compile(
            "(\\\\[a-zA-Z]+)|" + '((\\\\)*[$-/:-?{-~!"^_`\[\]])|' + "(\w)|" + "(\\\\)"
        )
        self.n_class = len(self.id2word)

    def tokenize(self, formula: str):
        tokens = re.finditer(self.TOKENIZE_PATTERN, formula)
        tokens = list(map(lambda x: x.group(0), tokens))
        tokens = [x for x in tokens if x is not None and x != ""]
        return tokens

class LatexPredictDataset(Dataset):
    def __init__(self, predict_list):
        super().__init__()
        # if predict_img_path:
        #     assert os.path.exists(predict_img_path), "Image not found"
        #     self.walker = predict_img_path
        # else:
        #     self.walker = ''
    
        self.transform = transforms.Compose([
            # transforms.PILToTensor()
            transforms.ToTensor()
        ])
        # self.img_list = glob.glob(predict_img_path + '\\*.jpg')
        self.img_list = predict_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]['img_name']
        image_array = np.array(self.img_list[idx]['pil'], dtype="uint8")
        image = Image.fromarray(image_array)
        # image = torchvision.io.read_image(img_path)
        image = self.transform(image)
        image = image.to(dtype=torch.float)
        # image /= image.max()
        # image = self.transform(image)  # transform image to [-1, 1]

        return image, img_name
    
    # def __getitem__(self, idx):
    #     img_name = self.img_list[idx]['img_name']
        
    #     im_b64 = self.img_list[idx]['pil']
    #     im_b64 = im_b64.encode('utf-8')
    #     im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
    #     im_file = BytesIO(im_bytes)  # convert image to file-like object
        
    #     image = Image.open(im_file)   # img is now PIL Image object
    #     image = self.transform(image)
    #     image = image.to(dtype=torch.float)

    #     return image, img_name
    
class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_set,
        val_set,
        test_set,
        predict_set,
        num_workers: int = 1,
        batch_size=20,
        text=None,
    ):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.predict_set = predict_set
        self.batch_size = batch_size
        self.text = text
        self.num_workers = num_workers

    def predict_dataloader(self):
        return DataLoader(self.predict_set, shuffle=False)
    
class ConvWithRowEncoder(nn.Module):
    def __init__(self, enc_dim: int):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0),
            nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512)
        )

        self.row_encoder = nn.LSTM(512, enc_dim, batch_first=True, bidirectional=True)

        self.enc_dim = enc_dim * 2  # bidirectional = True

    def forward(self, x: Tensor):
        """
            x: (bs, c, w, h)
        """
        conv_out = self.feature_encoder(x)  # (bs, c, w, h)
        conv_out = conv_out.permute(0, 2, 3, 1)  # (bs, w, h, c)

        bs, w, h, c = conv_out.size()
        rnn_out = []
        for row in range(w):
            row_data = conv_out[:, row, :, :]  # take a row data
            row_out, (h, c) = self.row_encoder(row_data)
            rnn_out.append(row_out)

        encoder_out = torch.stack(rnn_out, dim=1)
        bs, _, _, d = encoder_out.size()
        encoder_out = encoder_out.view(bs, -1, d)

        return encoder_out
    
class Attention(nn.Module):
    def __init__(self, enc_dim: int = 512, dec_dim: int = 512, attn_dim: int = 512):
        super().__init__()
        self.dec_attn = nn.Linear(dec_dim, attn_dim, bias=False)
        self.enc_attn = nn.Linear(enc_dim, attn_dim, bias=False)
        self.full_attn = nn.Linear(attn_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h: Tensor, V: Tensor):
        """
            input:
                h: (b, dec_dim) hidden state vector of decoder
                V: (b, w * h, enc_dim) encoder matrix representation
            output:
                context: (b, enc_dim)
        """

        attn_1 = self.dec_attn(h)
        attn_2 = self.enc_attn(V)
        attn = self.full_attn(torch.tanh(attn_1.unsqueeze(1) + attn_2)).squeeze(2)
        alpha = self.softmax(attn)
        context = (alpha.unsqueeze(2) * V).sum(dim=1)
        return context
    
class Decoder(nn.Module):
    def __init__(
        self,
        n_class: int,
        emb_dim: int = 80,
        enc_dim: int = 512,
        dec_dim: int = 512,
        attn_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        sos_id: int = 1,
        eos_id: int = 2,
    ):
        super().__init__()
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(n_class, emb_dim)
        self.attention = Attention(enc_dim, dec_dim, attn_dim)
        self.concat = nn.Linear(emb_dim + enc_dim, dec_dim)
        self.rnn = nn.LSTM(
            dec_dim,
            dec_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.rnn2 = nn.LSTM(
            dec_dim,
            dec_dim,
            num_layers=1,
            batch_first=True,
            bidirectional = bidirectional,
        )
        # self.layernorm = nn.LayerNorm((dec_dim))
        self.out = nn.Linear(dec_dim, n_class)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Embedding):
            nn.init.orthogonal_(layer.weight)
        elif isinstance(layer, nn.LSTM):
            for name, param in self.rnn.named_parameters():
                if name.startswith("weight"):
                    nn.init.orthogonal_(param)

    def forward(self, y, encoder_out=None, hidden_state=None):
        """
            input:
                y: (bs, target_len)
                h: (bs, dec_dim)
                V: (bs, enc_dim, w, h)
        """
        
        
        h, c = hidden_state
        embed = self.embedding(y)
        attn_context = self.attention(h, encoder_out)

        rnn_input = torch.cat([embed[:, -1], attn_context], dim=1)
        rnn_input = self.concat(rnn_input)

        rnn_input = rnn_input.unsqueeze(1)
        hidden_state = h.unsqueeze(0), c.unsqueeze(0)
        
        
        out, hidden_state = self.rnn(rnn_input, hidden_state)
        
        out = self.dropout(out)
        
        out, hidden_state = self.rnn2(out, hidden_state)
        # out = self.layernorm(out)
        out = self.logsoftmax(self.out(out))
        h, c = hidden_state
        return out, (h.squeeze(0), c.squeeze(0))
    
class Image2Latex(nn.Module):
    def __init__(
        self,
        n_class: int,
        enc_dim: int = 512,
        enc_type: str = "conv_row_encoder",
        emb_dim: int = 80,
        dec_dim: int = 512,
        attn_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        decode_type: str = "greedy",
        text: Text = None,
        beam_width: int = 5,
        sos_id: int = 1,
        eos_id: int = 2,
    ):
        assert enc_type in [
            "conv_row_encoder",
            "conv_encoder",
            "conv_bn_encoder",
            "resnet_encoder",
            "resnet_row_encoder",
        ], "Not found encoder"
        super().__init__()
        self.n_class = n_class

        self.encoder = ConvWithRowEncoder(enc_dim=enc_dim)
        enc_dim = self.encoder.enc_dim
        self.num_layers = num_layers
        self.decoder = Decoder(
            n_class=n_class,
            emb_dim=emb_dim,
            dec_dim=dec_dim,
            enc_dim=enc_dim,
            attn_dim=attn_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            sos_id=sos_id,
            eos_id=eos_id,
        )
        self.init_h = nn.Linear(enc_dim, dec_dim)
        self.init_c = nn.Linear(enc_dim, dec_dim)
        assert decode_type in ["greedy", "beamsearch"]
        self.decode_type = decode_type
        self.text = text
        self.beam_width = beam_width

    def init_decoder_hidden_state(self, V: Tensor):
        """
            return (h, c)
        """
        
        encoder_mean = V.mean(dim=1)
        h = torch.tanh(self.init_h(encoder_mean))
        c = torch.tanh(self.init_c(encoder_mean))
        return h, c

    def forward(self, x: Tensor, y: Tensor, y_len: Tensor):
        encoder_out = self.encoder(x)

        hidden_state = self.init_decoder_hidden_state(encoder_out)
 
        predictions = []
        for t in range(y_len.max().item()):
            dec_input = y[:, t].unsqueeze(1)
            out, hidden_state = self.decoder(dec_input, encoder_out, hidden_state)
            predictions.append(out.squeeze(1))

        predictions = torch.stack(predictions, dim=1)
        return predictions

    def decode(self, x: Tensor, max_length: int = 150):
        predict = []
        if self.decode_type == "greedy":
            predict = self.decode_greedy(x, max_length)
        elif self.decode_type == "beamsearch":
            predict = self.decode_beam_search(x, max_length)
        return self.text.int2text(predict)

    def decode_greedy(self, x: Tensor, max_length: int = 150):
        encoder_out = self.encoder(x)
        bs = encoder_out.size(0)

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        y = torch.LongTensor([self.decoder.sos_id]).view(bs, -1)

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        predictions = []
        for t in range(max_length):
            out, hidden_state = self.decoder(y, encoder_out, hidden_state)

            k = out.argmax().item()

            predictions.append(k)

            y = torch.LongTensor([k]).view(bs, -1)
        return predictions

    def decode_beam_search(self, x: Tensor, max_length: int = 150):
        """
            default: batch size equal to 1
        """
        encoder_out = self.encoder(x)
        bs = encoder_out.size(0)  # 1

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        list_candidate = [
            ([self.decoder.sos_id], hidden_state, 0)
        ]  # (input, hidden_state, log_prob)
        for t in range(max_length):
            new_candidates = []
            for inp, state, log_prob in list_candidate:
                y = torch.LongTensor([inp[-1]]).view(bs, -1).to(device=x.device)
                out, hidden_state = self.decoder(y, encoder_out, state)

                topk = out.topk(self.beam_width)
                new_log_prob = topk.values.view(-1).tolist()
                new_idx = topk.indices.view(-1).tolist()
                for val, idx in zip(new_log_prob, new_idx):
                    new_inp = inp + [idx]
                    new_candidates.append((new_inp, hidden_state, log_prob + val))

            new_candidates = sorted(new_candidates, key=lambda x: x[2], reverse=True)
            list_candidate = new_candidates[: self.beam_width]

        return list_candidate[0][0]

class Image2LatexModel(pl.LightningModule):
    def __init__(
        self,
        lr,
        total_steps,
        n_class: int,
        enc_dim: int = 512,
        enc_type: str = "conv_row_encoder",
        emb_dim: int = 80,
        dec_dim: int = 512,
        attn_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        decode_type: str = "greedy",
        text: Text = None,
        beam_width: int = 5,
        sos_id: int = 1,
        eos_id: int = 2,
        log_step: int = 100,
        log_text: bool = False,
    ):
        super().__init__()
        self.model = Image2Latex(
            n_class,
            enc_dim,
            enc_type,
            emb_dim,
            dec_dim,
            attn_dim,
            num_layers,
            dropout,
            bidirectional,
            decode_type,
            text,
            beam_width,
            sos_id,
            eos_id,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.total_steps = total_steps
        self.text = text
        self.max_length = 150
        self.log_step = log_step
        self.log_text = log_text
        self.save_hyperparameters()
        self.extract_result=[]
        

    def forward(self, images, formulas, formula_len):
        return self.model(images, formulas, formula_len)
    
    # Do things u want here at predict step
    def predict_step(self, batch, batch_idx):
        image, image_name = batch
        
        # image_name = img_path[0].split('\\')[-1]

        latex = self.model.decode(image, self.max_length)

        return image_name[0], latex
        # return image_name[0]

#################### CLASSIFICATION MATERIALS ####################
def classified_prediction(ckpt_path, device, img):
    mean = [0.9460, 0.9459, 0.9459]
    std = [0.1328, 0.1330, 0.1330]
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])
    
    img =  img.convert("RGB")
    img_transformed = test_transform(img)
    img_complete = torch.unsqueeze(img_transformed, 0)
    
    model = resnet18()
    num_of_features = model.fc.in_features
    num_of_classes = 2

    model.fc = nn.Linear(num_of_features, num_of_classes)
    checkpoint = torch.load(ckpt_path,  map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    
    model.eval()

    with torch.no_grad():
        output = model(img_complete.to(device))
        _, predicted = torch.max(output.data, 1)
    
    if predicted.item() == 1:
        return "single-line"
    else:
        return "multi-line"

def extracting(obj_list, ckpt_path):
    random_state=12
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    text = Text100k()
    
    emb_dim = 80
    dec_dim = 512
    enc_dim = 256
    attn_dim = 256
    
    lr = 0.001
    max_length = 150
    log_idx = 300
    # max_epochs = 15
    batch_size = 16
    # steps_per_epoch = round(len(train_set) / batch_size)
    # total_steps = steps_per_epoch * max_epochs
    num_workers = 4
    dm = DataModule(
        None,
        None,
        None,
        None,
        num_workers,
        batch_size,
        text,
    )
    
    num_layers = 1
    drop_out = 0.2
    decode = "beamsearch"
    beam_width=5
    
    model = Image2LatexModel(
        lr=lr,
        n_class=text.n_class,
        total_steps=1000,
        enc_dim=enc_dim,
        enc_type="conv_encoder",
        emb_dim=emb_dim,
        dec_dim=dec_dim,
        attn_dim=attn_dim,
        num_layers=num_layers,
        dropout=drop_out,
        sos_id=text.sos_id,
        eos_id=text.eos_id,
        decode_type="beamsearch",
        text=text,
        beam_width=beam_width,
        log_step=100,
        log_text="store_true",
    )
    
    grad_clip=3
    accumulate_batch=64
    max_epoch=15
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step") # type: ignore

    accumulate_grad_batches = accumulate_batch // batch_size
    trainer = pl.Trainer(
        callbacks=[lr_monitor],
        accelerator="auto",
        log_every_n_steps=1,
        gradient_clip_val=0,
    )
            
    dm.predict_set = LatexPredictDataset(obj_list)
    result = trainer.predict(datamodule=dm, model=model, ckpt_path=ckpt_path)
    
    print(result)
    
    return result

def predicting(pointList, predicted_img, classified_ckpt, extracted_ckpt):
    # Define type of device
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    image = Image.open(predicted_img)
    origin_img = image.convert("L")
    
    # add detected results to non_multi_list and multi_list
    detected_list = []
    non_multi_list = []
    multi_list = []
    index = 1

    for i in pointList:
        detected_sample = {
            "img_name": None,
            "pil": None,
            "coordinates": None,
            "class": None,
            "isolated": None,
            "latex": None
        }
        
        xmin, ymin, xmax, ymax = i[0], i[1], i[2], i[3] # coordinate of yolo bbox is [xmin, ymin, xmax, ymax]
        ### Alert: remeber to increase bbox Area to enhance extraction performance
        # crop_box = (int(xmin), int(ymin), int(xmax), int(ymax))
        crop_box = (int(xmin) - 5, int(ymin) - 5, int(xmax) + 5, int(ymax) + 5)
        im = origin_img.crop(crop_box)
        # save the image to predetermined savepath
        img_with_border = ImageOps.expand(im, border = 0, fill = 255) #add white border for crop image
        # img_with_border.save('crop_data/' + 'im_{}.jpg'.format(index))

        # detected_sample["img_name"] = 'im_{}.jpg'.format(index)
        detected_sample["img_name"] = i[6]
        # detected_sample["pil"] = img_with_border
        detected_sample["pil"] = np.array(img_with_border).tolist()
        detected_sample["coordinates"] = crop_box
        detected_sample["class"] = i[5]
        
        detected_list.append(detected_sample)
        
        if i[5] == "isolated":
            detected_sample["isolated"] = classified_prediction(classified_ckpt, device, img_with_border)
            if detected_sample["isolated"] == "multi-line":
                multi_list.append(detected_sample)
            else:
                non_multi_list.append(detected_sample)
        else:
            non_multi_list.append(detected_sample)
                
        index = index + 1
        
        
    # Extraction step for non-multi-line expr
    # split non_multi_list because if len(non_multi_list) too large, the request will not be loaded to cloud
    # split_non_multi_list = [non_multi_list[x:x+10] for x in range(0, len(non_multi_list), 10)] 
    # for i in split_non_multi_list:
    #     non_multi_results = predict_custom_trained_model_sample(
    #         project = project,
    #         endpoint_id = endpoint_id,
    #         location = location,
    #         instances = i,
    #         api_endpoint = api_endpoint
    #     )
    
    non_multi_results = extracting(non_multi_list, extracted_ckpt)

    # load extracted latex result into each non-multi-line expr
    for i in non_multi_list:
        for j in non_multi_results:
            if i['img_name'] == j[0]:
                i['latex'] = j[1]

                
    # Extraction step for multi-line expr
    for i in multi_list:
        latex_list = []
        latex_result = ""
        segmented_list = segment.segment_line(i['pil'])
        
        # latex_list = predict_custom_trained_model_sample(
        #     project = project,
        #     endpoint_id = endpoint_id,
        #     location = location,
        #     instances = segmented_list,
        #     api_endpoint = api_endpoint
        # )
        
        latex_list = extracting(segmented_list, extracted_ckpt)
               
        for j in latex_list:
            if latex_result == "":
                latex_result = j[1]
            else:
                latex_result = latex_result  + " \\\\ " + j[1]
                
        latex_result = "\\begin{aligned} " + latex_result + " \\end{aligned}"
        i["latex"] = latex_result
    
    # save all things to final result    
    final_result = multi_list + non_multi_list
    final_result = sorted(final_result, key=lambda x: int(x['img_name']))
    final_result = [{'img_name': i['img_name'],
                    'latex': i['latex']}
                    for i in final_result]
    return final_result
# def load_data(obj_list, ckpt_path):
#     extracting()

# if __name__ == '__main__':
#     text = Text100k()
#     # extracting(obj_list, ckpt_path, text)
    