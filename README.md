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

## Generating romantic captions 
