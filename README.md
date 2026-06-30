# Spot the Fake Photo - SalesCode AI Assignment

This project detects whether an input image is a real photo or a recaptured photo of another screen/printout.

The required command is supported:

```bash
python predict.py some_image.jpg
```

Output is one score from `0` to `1`:

- `0` = real photo
- `1` = screen/recapture photo

## Approach

I solved this as a lightweight image-forensics problem instead of object recognition. The detector extracts clues that commonly appear when a camera captures another display:

- moire / screen grid frequency peaks
- horizontal or vertical banding
- high-frequency texture artifacts
- glare and clipped highlights
- unnatural color quantization
- abnormal sharpness and edge patterns

A small logistic model is trained on these features using my collected `real/` and `screen/` images. If the trained model file is missing, `predict.py` still has a heuristic fallback, but the trained `recapture_model.json` should be used for final submission.

## Folder Structure

```text
salescode_fake_photo_detector/
├── predict.py
├── train.py
├── evaluate.py
├── app.py
├── requirements.txt
├── README.md
├── APPROACH_NOTE.md
├── recapture_model.json        # created after training
├── real/                       # add real photos here
├── screen/                     # add screen/printout photos here
└── sample_data/                # small demo dataset only
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Collection

For the final submission, collect at least:

- 50 to 100 real photos in `real/`
- 50 screen/recapture photos are included in `screen/`, replaced using the newly uploaded phone-captured screen dataset. More original phone-captured screen and printout photos can further improve hidden-test accuracy

Add variety: different screens, brightness levels, angles, indoor/outdoor lighting, glare, phone/laptop screens, and printouts.

## Train

```bash
python train.py --data .
```

This creates:

```text
recapture_model.json
```

## Evaluate

```bash
python evaluate.py --data .
```

The script reports:

- accuracy
- average latency
- P95 latency
- cost per image

## Predict One Image

```bash
python predict.py path/to/image.jpg
```

Example:

```bash
python predict.py screen/screen_001.jpg
```

## Live Demo

```bash
streamlit run app.py
```

The web page opens the camera, captures an image, runs `predict.py`, and shows the fraud/recapture score live.

## Cost

This solution can run locally/on-device and does not require any paid cloud API.

Estimated cost per image: approximately `₹0 / $0` on-device.


## Dataset included in this version

This version contains 50 real-class images created from the user's captured real-object photos using mild camera-style augmentations such as small rotation, crop, brightness/contrast changes and focus variation. The `screen/` folder has been replaced with 50 screen/recapture images from the uploaded screen dataset, with a few controlled variations where needed.

For best hidden-test performance, add more actual phone-captured images instead of relying only on augmentation. Real photos should be placed in `real/`; recaptured screen or printout photos should be placed in `screen/`.
