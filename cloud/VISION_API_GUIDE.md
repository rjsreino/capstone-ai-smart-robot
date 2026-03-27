# Vision API Guide - ChatGPT Camera Mode for OAK-D

This guide explains how to use OpenAI's Vision API with your OAK-D camera to analyze images, solve problems, and read text - just like ChatGPT's camera mode!

## What Can It Do?

Show your camera a piece of paper with:
- **Math problems**: "2+2=?" → Gets "The answer is 4"
- **Text**: Any written text → Gets it read/transcribed
- **Questions**: "What is AI?" → Gets comprehensive answer
- **Diagrams/Images**: Any image → Gets detailed description

## Setup

### 1. Install Dependencies

Already installed if you have the requirements.txt! The key packages are:
```bash
pip install openai>=1.0.0
pip install Pillow>=9.0.0
```

### 2. Set OpenAI API Key

```bash
# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"

# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Or add to .env file
echo 'OPENAI_API_KEY=your-api-key-here' >> .env
```

Get your API key from: https://platform.openai.com/api-keys

### 3. Camera Setup

Make sure your OAK-D or USB camera is connected:
```bash
# Check OAK-D
python -c "import depthai as dai; print(f'Found {len(dai.Device.getAllAvailableDevices())} OAK-D device(s)')"

# Check USB camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('USB camera OK' if cap.isOpened() else 'USB camera not found')"
```

## Quick Start

### Option 1: Quick Test Script

The fastest way to test:

```bash
cd cloud/examples
python quick_vision_test.py
```

This will:
1. Wait 5 seconds for you to position your paper
2. Capture image from camera
3. Send to OpenAI Vision API
4. Print the answer!

**Try it:**
- Write "2+2=?" on paper
- Run the script
- Hold paper to camera when countdown starts
- See the AI solve it!

### Option 2: Interactive Demo

For more options:

```bash
cd cloud/examples
python vision_demo.py
```

Choose from 6 different demos:
1. **Problem Solving** - Solve math/text problems
2. **Text Reading** - OCR any text
3. **Scene Description** - Describe what camera sees
4. **Custom Question** - Ask anything about the image
5. **Capture & Save** - Save image and analyze it
6. **Run All** - Try everything!

## Python API Usage

### Basic Example

```python
import asyncio
from vision import VisionProcessor

async def main():
    processor = VisionProcessor()
    
    try:
        # Solve a problem shown to camera
        answer = await processor.solve_problem()
        print(f"Answer: {answer}")
    finally:
        await processor.close()

asyncio.run(main())
```

### Advanced Examples

#### 1. Solve Math Problem

```python
from vision import VisionProcessor

processor = VisionProcessor()

# Show "2+2=?" to camera
solution = await processor.solve_problem()
# Returns: "The problem shows 2+2. The answer is 4."

await processor.close()
```

#### 2. Read Text (OCR)

```python
from vision import VisionProcessor

processor = VisionProcessor()

# Show any text to camera
text = await processor.read_text()
# Returns: "The text reads: [extracted text]"

await processor.close()
```

#### 3. Describe Scene

```python
from vision import VisionProcessor

processor = VisionProcessor()

# Point camera at anything
description = await processor.describe_scene()
# Returns: "I see a desk with a laptop, coffee mug, and notebook..."

await processor.close()
```

#### 4. Custom Question

```python
from vision import VisionProcessor

processor = VisionProcessor()

# Capture image
image = await processor.capture_image()

# Ask custom question
answer = await processor.analyze_image(
    image_bytes=image,
    question="What color is the object in the center?",
    max_tokens=200
)

await processor.close()
```

#### 5. Analyze Existing Image File

```python
from vision import VisionProcessor

processor = VisionProcessor()

# Load image from file
with open("problem.jpg", "rb") as f:
    image_bytes = f.read()

# Analyze it
answer = await processor.solve_problem(image_bytes)
print(answer)

await processor.close()
```

#### 6. Capture and Save

```python
from vision import VisionProcessor

processor = VisionProcessor()

# Capture from camera
image = await processor.capture_image()

# Save it
processor.save_image(image, "captured.jpg")

# Analyze it
result = await processor.analyze_image(image, "What is this?")

await processor.close()
```

## REST API Endpoints

The vision functionality is integrated into the FastAPI server.

### Start the Server

```bash
cd cloud
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API Documentation

Visit: http://localhost:8000/docs

### Endpoint 1: Capture and Analyze

**POST** `/vision/capture-and-analyze`

Capture image from camera and analyze with custom question.

```bash
# Example: Solve a problem
curl -X POST "http://localhost:8000/vision/capture-and-analyze" \
  -F "question=What is written on this paper? If it's a problem, solve it." \
  -F "save_image=false"

# Response:
{
  "response": "The paper shows 2+2. The answer is 4.",
  "success": true
}
```

```bash
# Example: With image saved
curl -X POST "http://localhost:8000/vision/capture-and-analyze" \
  -F "question=Describe what you see" \
  -F "save_image=true"

# Response includes base64 image
{
  "response": "I see...",
  "success": true,
  "image_base64": "/9j/4AAQSkZJRg..."
}
```

### Endpoint 2: Solve Problem

**POST** `/vision/solve-problem`

Optimized for solving problems (math, puzzles, questions).

```bash
curl -X POST "http://localhost:8000/vision/solve-problem" \
  -F "save_image=false"

# Response:
{
  "solution": "The problem asks for 2+2. Let me solve: 2 + 2 = 4. The answer is 4.",
  "success": true
}
```

### Endpoint 3: Read Text (OCR)

**POST** `/vision/read-text`

Extract and read all text from image.

```bash
curl -X POST "http://localhost:8000/vision/read-text" \
  -F "save_image=false"

# Response:
{
  "text": "Hello World\nThis is a test\n...",
  "success": true
}
```

## JavaScript/TypeScript Usage

For mobile app or web frontend:

```typescript
// Capture and analyze
async function solveFromCamera() {
  const response = await fetch('http://localhost:8000/vision/solve-problem', {
    method: 'POST',
    body: new FormData()
  });
  
  const data = await response.json();
  console.log('Solution:', data.solution);
}

// With custom question
async function analyzeWithQuestion(question: string) {
  const formData = new FormData();
  formData.append('question', question);
  formData.append('save_image', 'false');
  
  const response = await fetch('http://localhost:8000/vision/capture-and-analyze', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  console.log('Response:', data.response);
}

// Use it
await solveFromCamera();
await analyzeWithQuestion('What color is this?');
```

## Configuration

### Change Vision Model

By default uses `gpt-4o`. You can change it:

```python
from vision import VisionProcessor

# Use different model
processor = VisionProcessor(vision_model="gpt-4-turbo")

# Or set in config.py
OPENAI_VISION_MODEL = "gpt-4o"  # or "gpt-4-turbo", "gpt-4-vision-preview"
```

### Choose Camera

```python
from vision import VisionProcessor

# Use OAK-D (default)
processor = VisionProcessor(use_oakd=True)

# Use USB camera instead
processor = VisionProcessor(use_oakd=False)
```

## Tips for Best Results

### Lighting
- ✅ Good lighting improves accuracy
- ✅ Avoid shadows on the paper
- ✅ No glare or reflections

### Positioning
- ✅ Hold paper steady and flat
- ✅ Fill most of the camera view
- ✅ Keep text/problem clearly visible
- ❌ Don't position too far or too close

### Writing
- ✅ Write clearly and large enough
- ✅ Use dark ink on light paper (high contrast)
- ✅ Print text works best
- ❌ Avoid messy handwriting

### Problems to Solve
Works great for:
- ✅ Simple math: "2+2=?", "5×3=?"
- ✅ Complex math: "x² + 2x - 8 = 0"
- ✅ Word problems: "If John has 3 apples..."
- ✅ Reading comprehension questions
- ✅ Any text that needs transcription

## Troubleshooting

### "Camera not available"
```bash
# Check camera connection
python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"

# Try USB camera instead
processor = VisionProcessor(use_oakd=False)
```

### "OpenAI API key not found"
```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."

# Verify it's set
echo $OPENAI_API_KEY

# Or pass directly in code
processor = VisionProcessor(openai_api_key="sk-...")
```

### "Vision API returned no response"
- Check your internet connection
- Verify API key is valid
- Check API usage/billing at platform.openai.com
- Try with better lighting
- Ensure something is visible to camera

### "Failed to capture image"
- Check camera is connected
- Try different camera: `VisionProcessor(use_oakd=False)`
- Check USB connection
- Restart the device

### Import Errors
```bash
# Make sure you're in the right directory
cd cloud

# Or add to Python path
import sys
sys.path.insert(0, '/path/to/server/cloud')

from vision import VisionProcessor
```

## Cost Information

OpenAI Vision API pricing (as of 2024):
- **gpt-4o**: ~$0.01 per image (1024×1024)
- **gpt-4-turbo**: ~$0.01 per image
- Much cheaper than gpt-4-vision-preview

Each capture+analyze call = 1 API request.

## Examples Gallery

### 1. Math Problem
**Show:** Paper with "2+2=?"
**Get:** "The problem shows 2+2. The answer is 4."

### 2. Equation
**Show:** "x² - 4 = 0"
**Get:** "To solve x² - 4 = 0, we factor: (x-2)(x+2) = 0, so x = 2 or x = -2"

### 3. Text Document
**Show:** Any printed or handwritten text
**Get:** Full transcription of the text

### 4. Question
**Show:** "What is artificial intelligence?"
**Get:** Comprehensive answer explaining AI

### 5. Diagram
**Show:** Any diagram, chart, or image
**Get:** Detailed description of what's shown

## Next Steps

1. **Try the quick test**: `python examples/quick_vision_test.py`
2. **Integrate into your app**: Use REST API or Python API
3. **Customize prompts**: Adjust questions for your use case
4. **Add to robot commands**: "Hey Rovy, what do you see?" → captures and describes

## Need Help?

- Check the example scripts in `cloud/examples/`
- Read the inline code documentation
- Test with `quick_vision_test.py` first
- Verify camera and API key setup

Enjoy your ChatGPT camera mode! 🎥✨

