neural_style_synthesizer
==========================

INSTALL
---------------

The model files of neural networks are not contained in this repository.
You can get them from [nin_imagenet.caffemodel](https://gist.github.com/mavenlin/d802a5849de39225bcc6) and [VGG_ILSVRC_16_layers.caffemodel](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md).

Dependent libraries are installed with the following script.

```
pip install numpy
pip install -r requirements.txt
```

RUN
----------------

### Whole style transfer

You can transfer whole patch from one to another.

with CPU

```
python bin/convert_image_multi.py \
  --modelpath=./VGG_ILSVRC_16_layers.caffemodel \
  --iteration=100 \
  --gpu=-1 \
  --xsplit=1 --ysplit=1 --resize=300 \
  input.png \
  style.png \
  --output_image=./converted.png
```

with GPU

```
python bin/convert_image_multi.py \
  --modelpath=./VGG_ILSVRC_16_layers.caffemodel \
  --iteration=100 \
  --gpu=0 \
  --xsplit=1 --ysplit=1 --resize=300 \
  input.png \
  style.png \
  --output_image=./converted.png
```

### Partial style transfer

Choose optimal patches from style image and transfer them to another image.
Split style image to 2x2

```
python bin/convert_image_multi.py \
  --modelpath=./VGG_ILSVRC_16_layers.caffemodel \
  --iteration=100 \
  --gpu=0 \
  --xsplit=2 --ysplit=2 --resize=300 \
  --model=vgg_nopad\
  input.png \
  style.png \
  --output_image=./converted_optimal_2x2.png
```

### Style transferred video

Tranfer style on video frame using last frame's result.

```
python bin/convert_video.py \
  --iteration=100 --model=vgg \
  video.mp4 \
  style.png \
  output_directory
```

Then you can find the style transferred video at `output_directory/out.avi` after 100 x frame times calculation.

### Optimal Blended Texture Transfer

Please see https://nico-opendata.jp/en/casestudy/neural_style_synthesizer/index.html for technical details.

```
python bin/convert_image_multistyle.py \
  --model=vgg_nopad \
  --iteration=100 \
  --gpu=3 --xsplit=1 --ysplit=1 --resize=200 \
  /path/to/input/file \
  /path/to/directory/contains/multiple/refarence/files \
  --debug --out_dir=/path/of/output
```
