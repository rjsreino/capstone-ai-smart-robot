# Vision API Examples

This folder contains examples and demos for using the Vision API with OAK-D camera.

## Quick Start

### 1. Verify Setup

First, check that everything is configured correctly:

```bash
python verify_vision_setup.py
```

This will check:
- ✅ OpenAI library installed
- ✅ API key configured
- ✅ Camera available
- ✅ All dependencies working

### 2. Quick Test

The fastest way to test the vision functionality:

```bash
python quick_vision_test.py
```

**What it does:**
- Counts down 5 seconds
- Captures image from camera
- Sends to OpenAI Vision API
- Prints the answer

**Try it:**
1. Write "2+2=?" on paper
2. Run the script
3. Hold paper to camera during countdown
4. See the AI solve it!

### 3. Interactive Demo

For a full-featured demo with multiple options:

```bash
python vision_demo.py
```

**Features:**
1. **Problem Solving** - Show math/text problems and get solutions
2. **Text Reading** - OCR any text from camera
3. **Scene Description** - Describe what the camera sees
4. **Custom Question** - Ask anything about an image
5. **Capture & Save** - Save the image and analyze it
6. **Run All** - Try all demos in sequence

## Files

| File | Description |
|------|-------------|
| `verify_vision_setup.py` | Check if all requirements are met |
| `quick_vision_test.py` | Quick 5-second test - solve a problem |
| `vision_demo.py` | Interactive demo with multiple options |

## Requirements

- Python 3.8+
- OpenAI API key set in environment: `export OPENAI_API_KEY="sk-..."`
- Camera connected (OAK-D or USB webcam)
- Dependencies installed (see `../requirements.txt`)

## Usage Tips

### For Best Results

**Lighting:**
- Use good lighting (no shadows)
- Avoid glare/reflections
- Natural or bright white light works best

**Positioning:**
- Hold paper steady and flat
- Fill most of the camera view
- Keep text clearly visible
- Not too close, not too far

**Writing:**
- Write clearly and large enough
- Use dark ink on light paper (high contrast)
- Print text works better than handwriting
- Avoid messy or tiny writing

### What Works Well

✅ Simple math: "2+2=?", "5×3=?"
✅ Complex math: "x² + 2x - 8 = 0"
✅ Word problems: "If John has 3 apples and gives 1 to Mary..."
✅ Text transcription: Any readable text
✅ Questions: "What is artificial intelligence?"
✅ Object recognition: Show an object and ask "What is this?"

## Troubleshooting

### Camera Not Found

```bash
# Check OAK-D
python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"

# Check USB camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'Not found')"

# If OAK-D not working, try USB camera:
# Edit the script and set: use_oakd=False
```

### API Key Not Set

```bash
# Linux/Mac
export OPENAI_API_KEY="sk-your-key-here"

# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"

# Or create .env file in cloud/ directory
echo 'OPENAI_API_KEY=sk-your-key-here' > ../.env
```

### Import Errors

Make sure you're in the right directory:

```bash
# Should be in cloud/examples/
pwd
# Output: .../server/cloud/examples

# If not, navigate there
cd /path/to/server/cloud/examples
```

### Vision API Errors

- **"API key not found"**: Set `OPENAI_API_KEY` environment variable
- **"Rate limit"**: Wait a bit, you've made too many requests
- **"No response"**: Check internet connection and API key validity
- **"Failed to capture"**: Check camera connection

## Example Output

```
============================================================
  🎥 QUICK VISION TEST
============================================================

📝 Instructions:
   1. Write something on paper (e.g., '2+2=?', 'What is AI?', etc.)
   2. Hold it up to the camera
   3. Get AI analysis!

⏳ Starting in 5 seconds...
   (Position your paper now!)

   5...
   4...
   3...
   2...
   1...

📸 Capturing and analyzing...

✅ AI Response:
======================================================================
The image shows a handwritten math problem: "2+2=?"

The answer is: 4

To solve: 2 + 2 = 4
======================================================================

✨ Success!
```

## Next Steps

1. ✅ Run `verify_vision_setup.py` to check your setup
2. ✅ Try `quick_vision_test.py` with a simple problem
3. ✅ Explore `vision_demo.py` for more features
4. ✅ Read `../VISION_API_GUIDE.md` for full documentation
5. ✅ Integrate into your app using the REST API or Python module

## Support

- **Full Guide**: See `../VISION_API_GUIDE.md`
- **Python API**: See `../vision.py` for code documentation
- **REST API**: Start server and visit http://localhost:8000/docs
- **Issues**: Check troubleshooting section above

Enjoy! 🎉

