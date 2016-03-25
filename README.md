chainer-neural-art
==================

INSTALL
---------------

```
wget
pip install -r requirements.txt
```

RUN
----------------

with CPU

```
mkdir output
python bin/convert_image_multi.py \
  --iteration=1000 \
  --gpu=-1 \
  --xsplit=1 --ysplit=1 --resize=300 \
  input.png \
  style.png \
  --out_dir=output
```

with GPU

```
mkdir output
python bin/convert_image_multi.py \
  --iteration=1000 \
  --gpu=0 \
  --xsplit=1 --ysplit=1 --resize=300 \
  input.png \
  style.png \
  --out_dir=output
```
