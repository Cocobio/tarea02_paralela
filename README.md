Prepare image data:

```bash
[[ -d "data" ]] || mkdir "data"
cd data
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
unzip DIV2K_valid_LR_bicubic_X4.zip
rm -f DIV2K_valid_LR_bicubic_X4.zip
cd -
```
