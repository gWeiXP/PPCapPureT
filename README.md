# PPCap_PureT
# Using the style discriminator to guide the factual model in generating stylized captions.
Run generate_guide.py to apply PPCap to PureT and generate stylized captions.
Important arguments include:
* `--pretrained_path`, the path of pre-trained factual model
* `--gedi_model_name_or_path`, the path of trained factual model
* `--code_1`, the desired style
* `--code_0`, the undesired style
* `--disc_weight`, the weight w
* `--data_test`, the path of test set
* `--teststyle`, the desired style
* `--gen_model_type`, the type of factual model

## Factual model
If using PureT-XE as the factual model, set `--pretrained_path` to the path where `'model_pureT_XE_16.pth'` is located, and set `--gen_model_type` to `'pureT_XE'`. If using PureT-SCST as the factual model, set `--pretrained_path` to the path where `'model_pureT_SCST_30.pth'` is located, and set `--gen_model_type` to `'pureT_SCST'`.

## Test on SentiCap dataset
* `--data_test` needs to be set to the path where `'Senticap_ViT-L_14_test.pkl'` is located.
* If generating positive captions, set `--gedi_model_name_or_path` to the path where `'model_pos_9.pt'` is located,  `--code_1` to `'positive'`, `--code_0` to `'negative'`, `--disc_weight` to `200`, `--teststyle` to `'positive'`.
* If generating negative captions, set `--gedi_model_name_or_path` to the path where `'model_neg_9.pt'` is located,  `--code_1` to `'negative'`, `--code_0` to `'positive'`, `--disc_weight` to `175`, `--teststyle` to `'negative'`.

## Test on FlickrStyle10k dataset
* `--data_test` needs to be set to the path where `'FlickrStyle10k_ViT-L_14_test.pkl'` is located.
* If generating romantic captions, set `--gedi_model_name_or_path` to the path where `'model_ro_1.pt'` is located,  `--code_1` to `' romantic'`, `--code_0` to `' factual'`, `--disc_weight` to `140`, `--teststyle` to `'romantic'`.
* If generating humorous captions, set `--gedi_model_name_or_path` to the path where `'model_fu_1.pt'` is located,  `--code_1` to `' humorous'`, `--code_0` to `' factual'`, `--disc_weight` to `175`, `--teststyle` to `'humorous'`.
* Ensure there is a space at the beginning of `--code_1` and `--code_1`.
