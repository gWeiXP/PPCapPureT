# PPCap_PureT
# Using the style discriminator to guide the factual model in generating stylized captions.
Run generate_guide.py to apply PPCap to PureT and generate stylized captions.
Important arguments include:
* `--pretrained_path`, the path of pre-trained factual model
* `--gedi_model_name_or_path`, the path of trained stylized model
* `--code_1`, the desired style
* `--code_0`, the undesired style
* `--disc_weight`, the weight w
* `--data_test`, the path of test set
* `--teststyle`, the desired style
* `--gen_model_type`, the type of factual model

Apologies, we noticed that there are some absolute paths in the code that need to be modified to match the file paths on your device.
* The paths like "/home/liwc/wxp/refercode/GeDi_Final/PPL/LM_ro" in `utils.py eval_ppl()` : we have uploaded the `PPL` folder to [PPCap](https://github.com/gWeiXP/PPCap).
* The paths like "/home/liwc/wxp/dataset/MSCOCO/train2014/" in `utils.py ClipCocoDataset()` : you may need to modify the code so that `filename` obtains the paths of images on your device.
* The paths ""/home/liwc/wxp/refercode/DataTestProcess/bert-base-uncased/vocab.txt"" in `utils.py eval_acc()` : we have uploaded the `bert-base-uncased` folder to [PPCap](https://github.com/gWeiXP/PPCap).

## Datasets and trained models
* Download the processed dataset and trained models from [Baidu Netdisk](https://pan.baidu.com/s/1rBb8-4_lp2IfxJkEO0MmmA). the extracted code is 'zp8c'. The 'classifier' folder needs to be placed in the current directory.
* Download MSCOCO [validation images](http://images.cocodataset.org/zips/val2014.zip)
* Please make sure to modify the code for image paths in the ClipCocoDataset class within the utils.py file to obtain the correct image paths.
* Regarding the environment, please note that `transformers==2.8.0` is required; higher versions may cause incompatibility issues


## Factual model
If using PureT-XE as the factual model, set `--pretrained_path` to the path where `'model_pureT_XE_16.pth'` is located, and set `--gen_model_type` to `'pureT_XE'`. If using PureT-SCST as the factual model, set `--pretrained_path` to the path where `'model_pureT_SCST_30.pth'` is located, and set `--gen_model_type` to `'pureT_SCST'`.

## Test on SentiCap dataset
* `--data_test` needs to be set to the path where `'Senticap_ViT-L_14_test.pkl'` is located.
* If generating positive captions, set `--gedi_model_name_or_path` to the path where `'model_pos_9.pt'` is located,  `--code_1` to `'positive'`, `--code_0` to `'negative'`, `--disc_weight` to `200`, `--teststyle` to `'positive'`.
* If generating negative captions, set `--gedi_model_name_or_path` to the path where `'model_neg_9.pt'` is located,  `--code_1` to `'negative'`, `--code_0` to `'positive'`, `--disc_weight` to `175`, `--teststyle` to `'negative'`.

## Test on FlickrStyle10k dataset
* `--data_test` needs to be set to the path where `'FlickrStyle10k_ViT-L_14_test.pkl'` is located.
* If generating romantic captions, set `--gedi_model_name_or_path` to the path where `'model_ro_1.pt'` is located, (Ensure there is a space at the beginning of `--code_1` and `--code_0`) `--code_1` to `' romantic'`, `--code_0` to `' factual'`, `--disc_weight` to `140`, `--teststyle` to `'romantic'`.
* If generating humorous captions, set `--gedi_model_name_or_path` to the path where `'model_fu_1.pt'` is located, (Ensure there is a space at the beginning of `--code_1` and `--code_0`) `--code_1` to `' humorous'`, `--code_0` to `' factual'`, `--disc_weight` to `175`, `--teststyle` to `'humorous'`.

## Examples on our device
* `positive:`generate parameters %s Namespace(batch_size=36, code_0='negative', code_1='positive', data_test='/home/liwc/wxp/Alignment/github/dataset/Senticap/Senticap_ViT-L_14_test.pkl', device='cuda:1', disc_weight=200, gedi_model_name_or_path='/home/liwc/wxp/Alignment/github/trained_model/stylized_model/model_pos_9.pt', gen_model_type='pureT_SCST', generated_path='./generated/guide', max_seq_len=17, pretrained_path='/home/liwc/wxp/Alignment/github/trained_model/factual_model/model_pureT_SCST_30.pth', teststyle='positive', vocab_path='./mscoco/txt/coco_vocabulary.txt')
* `positive result:`{"clipscore": 0.6171875, "refclipscores": 0.68310546875, "bleu": [0.6055224948014654, 0.40025514673446183, 0.26998523921352235, 0.18244130589230556], "meteor": 0.21552771744253116, "rouge": 0.4513981632409327, "cider": 0.9324792714333866, "ppl": 15.86389, "acc": 0.9687964338781575}
* `negative:`generate parameters %s Namespace(batch_size=36, code_0='positive', code_1='negative', data_test='/home/liwc/wxp/Alignment/github/dataset/Senticap/Senticap_ViT-L_14_test.pkl', device='cuda:1', disc_weight=175, gedi_model_name_or_path='/home/liwc/wxp/Alignment/github/trained_model/stylized_model/model_neg_9.pt', gen_model_type='pureT_SCST', generated_path='./generated/guide', max_seq_len=17, pretrained_path='/home/liwc/wxp/Alignment/github/trained_model/factual_model/model_pureT_SCST_30.pth', teststyle='negative', vocab_path='./mscoco/txt/coco_vocabulary.txt')
* `negative result:`{"clipscore": 0.59326171875, "refclipscores": 0.650390625, "bleu": [0.5674791009853676, 0.3697376824178781, 0.2540249166351839, 0.17291863406810362], "meteor": 0.1940519525060811, "rouge": 0.4188755353978901, "cider": 0.8732718129002723, "ppl": 17.23189, "acc": 0.9662027833001988}
* `romantic:`generate parameters %s Namespace(batch_size=36, code_0=' factual', code_1=' romantic', data_test='/home/liwc/wxp/Alignment/github/dataset/FlickrStyle10k/FlickrStyle10k_ViT-L_14_test.pkl', device='cuda:1', disc_weight=140, gedi_model_name_or_path='/home/liwc/wxp/Alignment/github/trained_model/stylized_model/model_ro_1.pt', gen_model_type='pureT_SCST', generated_path='./generated/guide', max_seq_len=17, pretrained_path='/home/liwc/wxp/Alignment/github/trained_model/factual_model/model_pureT_SCST_30.pth', teststyle='romantic', vocab_path='./mscoco/txt/coco_vocabulary.txt')
* `romantic result:`{"clipscore": 0.58544921875, "refclipscores": 0.59326171875, "bleu": [0.2287594330455075, 0.10423137128501891, 0.050195192588284546, 0.024304886107113104], "meteor": 0.09808024347661877, "rouge": 0.2437544937587507, "cider": 0.30197201294467013, "ppl": 39.23921, "acc": 0.912}
* `humorous:`generate parameters %s Namespace(batch_size=36, code_0=' factual', code_1=' humorous', data_test='/home/liwc/wxp/Alignment/github/dataset/FlickrStyle10k/FlickrStyle10k_ViT-L_14_test.pkl', device='cuda:1', disc_weight=175, gedi_model_name_or_path='/home/liwc/wxp/Alignment/github/trained_model/stylized_model/model_fu_1.pt', gen_model_type='pureT_SCST', generated_path='./generated/guide', max_seq_len=17, pretrained_path='/home/liwc/wxp/Alignment/github/trained_model/factual_model/model_pureT_SCST_30.pth', teststyle='humorous', vocab_path='./mscoco/txt/coco_vocabulary.txt')
* `humorous result:`{"clipscore": 0.57763671875, "refclipscores": 0.57177734375, "bleu": [0.22687633400488066, 0.09754648145445417, 0.04129105583201404, 0.01725161244359373], "meteor": 0.09284519849801177, "rouge": 0.22646387187823705, "cider": 0.26090827971105135, "ppl": 48.37523, "acc": 0.804}
