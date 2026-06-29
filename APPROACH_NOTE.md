# Approach Note - Spot the Fake Photo

I treated this task as an image-forensics problem rather than a normal object-recognition problem. The goal is not to identify the object in the image, but to identify subtle artifacts that appear when a camera captures another phone/laptop screen or a printout.

My solution extracts lightweight computer-vision features from each image. These include frequency-domain peaks for moire/screen-grid artifacts, horizontal and vertical banding, high-frequency texture ratio, glare and highlight clipping, border-edge behavior, color quantization, saturation statistics, sharpness, and gradient texture. These features are then calibrated using a small logistic model trained on the collected `real/` and `screen/` images. The final script `predict.py` outputs one probability score from 0 to 1, where 0 means real photo and 1 means photo of a screen/recapture.

The method is intentionally small and fast. It does not use a heavy cloud model or external API, so it can potentially run on-device inside a mobile app.

## Results

Cross-validation accuracy: 96.00%
Evaluation accuracy: 100.00% on collected dataset
Average latency: 10.24 ms/image on MacBook Air CPU
P95 latency: 12.08 ms/image
Cost per image: ₹0 / $0 because it runs on-device

## Cut-off Score

The default threshold is 0.50. If the business goal is to catch more fraud, I would lower the threshold slightly to improve recall. If the business goal is to avoid wrongly blocking genuine users, I would increase the threshold to improve precision.

## Future Improvements

With more time, I would collect more hard cases: real photos containing monitors, glossy surfaces, glass reflections, low-light images, matte screens, high-refresh screens, cracked screens, and printouts under different lighting. I would also keep improving the dataset as cheaters adapt, and periodically recalibrate the threshold using fresh production examples.


Dataset update: I added 50 real-class samples made from my own phone-captured real-object images using multiple angles/crops and mild image variations. I also added actual screen-capture examples for the recapture class. Final hidden accuracy can be improved further by collecting more physical photos from more phones, screens, lighting conditions and printouts.


## Dataset used in this package

This packaged version contains 50 real images and 50 screen/recapture images. The screen class uses the newly uploaded phone-captured screen images. Some uploaded HEIC files could not be used directly by the Python/OpenCV runtime, so the final package uses converted JPG screen images plus a few controlled variations to keep exactly 50 screen samples. For the strongest hidden-test version, I would still add more original screen photos from multiple devices and printouts.
