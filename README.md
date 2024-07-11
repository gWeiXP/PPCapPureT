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

## factual model
If using PureT-XE as the factual model, set `--pretrained_path` to the path where `'model_pureT_XE_16.pth'` is located, and set `--gen_model_type` to `'pureT_XE'`. If using PureT-SCST as the factual model, set `--pretrained_path` to the path where `'model_pureT_SCST_30.pth'` is located, and set `--gen_model_type` to `'pureT_SCST'`.
