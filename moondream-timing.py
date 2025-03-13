import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Configuration constants
IMAGE_PATH = "image.webp"  # Change this to your image path

# Enable GPU memory stats if using CUDA
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()

print("Loading model...")
start_load = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"": "cuda"} if torch.cuda.is_available() else None
)
end_load = time.time()
print(f"Model loading time: {end_load - start_load:.2f} seconds\n")

# Image loading and encoding
print("Loading and encoding image...")
start_img_load = time.time()
image = Image.open(IMAGE_PATH)
img_load_time = time.time() - start_img_load
print(f"Image loading time: {img_load_time:.4f} seconds")

# Encode image and measure time
start_encoding = time.time()
encoded_image = model.encode_image(image)
encoding_time = time.time() - start_encoding
print(f"Image encoding time: {encoding_time:.4f} seconds\n")

# 1. Test Short Caption with encoded image
print("1. Short caption:")
start_short = time.time()
short_caption = model.caption(encoded_image, length="short")["caption"]
short_time = time.time() - start_short
print(f"{short_caption}")
print(f"Short caption generation time: {short_time:.4f} seconds\n")

# 2. Test Detailed Caption with encoded image
print("2. Detailed caption:")
start_detailed = time.time()
detailed_caption = model.caption(encoded_image, length="normal")["caption"]
detailed_time = time.time() - start_detailed
print(f"{detailed_caption}")
print(f"Detailed caption generation time: {detailed_time:.4f} seconds\n")

# Test streaming (timed separately as it's a different use case)
print("3. Streaming caption (timing not representative due to output):")
start_stream = time.time()
for t in model.caption(encoded_image, length="normal", stream=True)["caption"]:
    print(t, end="", flush=True)
stream_time = time.time() - start_stream
print(f"\nStreaming caption total time: {stream_time:.4f} seconds\n")

# 3. Visual Question Answering
print("4. Visual Question Answering:")
start_qa = time.time()
answer = model.query(encoded_image, "How many people are in the image?")["answer"]
qa_time = time.time() - start_qa
print(f"Q: How many people are in the image?")
print(f"A: {answer}")
print(f"VQA time: {qa_time:.4f} seconds\n")

# 4. Object Detection
print("5. Object Detection:")
start_detect = time.time()
objects = model.detect(encoded_image, "face")["objects"]
detect_time = time.time() - start_detect
print(f"Found {len(objects)} face(s)")
print(f"Object detection time: {detect_time:.4f} seconds\n")

# 5. Visual Pointing
print("6. Visual Pointing:")
start_point = time.time()
points = model.point(encoded_image, "person")["points"]
point_time = time.time() - start_point
print(f"Found {len(points)} person(s)")
print(f"Visual pointing time: {point_time:.4f} seconds\n")

# Summary of timing results
print("=========== TIMING SUMMARY ===========")
print(f"Model loading time: {end_load - start_load:.2f} seconds")
print(f"Image loading time: {img_load_time:.4f} seconds")
print(f"Image encoding time: {encoding_time:.4f} seconds")
print(f"Short caption time: {short_time:.4f} seconds")
print(f"Detailed caption time: {detailed_time:.4f} seconds")
print(f"Streaming caption time: {stream_time:.4f} seconds")
print(f"VQA time: {qa_time:.4f} seconds")
print(f"Object detection time: {detect_time:.4f} seconds")
print(f"Visual pointing time: {point_time:.4f} seconds")

# Calculate total inference time (excluding model loading)
total_inference = encoding_time + short_time + detailed_time + qa_time + detect_time + point_time
print(f"Total inference time: {total_inference:.4f} seconds")

# Print GPU memory usage if available
if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory usage: {(peak_memory - start_memory) / 1024**2:.2f} MB")
