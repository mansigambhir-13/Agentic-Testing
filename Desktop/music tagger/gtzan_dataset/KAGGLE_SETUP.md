# Kaggle API Setup Guide

This guide will help you set up Kaggle API credentials to download the GTZAN dataset.

## 🔑 Required Credentials

To download datasets from Kaggle, you need:

1. **Kaggle Account** (free)
2. **API Token** (kaggle.json file)

## 📝 Step-by-Step Setup

### Step 1: Create Kaggle Account

1. Go to [https://www.kaggle.com](https://www.kaggle.com)
2. Click "Sign Up" or "Register"
3. Complete the registration process (it's free!)

### Step 2: Get Your API Token

1. Log in to your Kaggle account
2. Click on your profile picture (top right)
3. Go to **"Account"** from the dropdown menu
4. Scroll down to the **"API"** section
5. Click **"Create New API Token"**
6. This will download a file named `kaggle.json`

### Step 3: Place the Credentials File

#### On Windows:

1. Create the `.kaggle` folder in your user directory:
   ```
   C:\Users\YourUsername\.kaggle\
   ```

2. Move the downloaded `kaggle.json` file to this folder:
   ```
   C:\Users\YourUsername\.kaggle\kaggle.json
   ```

#### On Linux/Mac:

1. Create the `.kaggle` folder in your home directory:
   ```bash
   mkdir ~/.kaggle
   ```

2. Move the downloaded `kaggle.json` file:
   ```bash
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   ```

3. Set proper permissions (IMPORTANT):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 4: Verify Setup

Run the setup script to verify:

```bash
python setup.py
```

Or test manually:

```python
import kagglehub
dataset_path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
print(f"Dataset downloaded to: {dataset_path}")
```

## 📋 What's in kaggle.json?

The `kaggle.json` file contains your API credentials:

```json
{
  "username": "your_username",
  "key": "your_api_key_here"
}
```

**⚠️ IMPORTANT:** Never share this file or commit it to version control!

## 🔒 Security Notes

1. **Never commit `kaggle.json` to Git**
   - Add it to `.gitignore`:
     ```
     .kaggle/kaggle.json
     ```

2. **Keep your API key private**
   - Don't share it publicly
   - Don't include it in code or documentation

3. **Set proper file permissions** (Linux/Mac)
   - Only you should be able to read the file
   - Use: `chmod 600 ~/.kaggle/kaggle.json`

## ❌ Troubleshooting

### Error: "Could not find kaggle.json"

**Solution:**
- Verify the file is in the correct location:
  - Windows: `C:\Users\YourUsername\.kaggle\kaggle.json`
  - Linux/Mac: `~/.kaggle/kaggle.json`
- Check the file name is exactly `kaggle.json` (not `kaggle.json.txt`)

### Error: "Permission denied" (Linux/Mac)

**Solution:**
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Error: "Invalid API credentials"

**Solution:**
- Regenerate your API token from Kaggle
- Make sure you copied the entire key (it's long!)
- Check for extra spaces or line breaks

### Error: "Dataset not found"

**Solution:**
- Make sure you've accepted the dataset's terms of use on Kaggle
- Go to the dataset page and click "Download" once to accept terms
- The dataset URL is: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

## 🔄 Regenerating API Token

If you need to regenerate your API token:

1. Go to Kaggle → Account → API
2. Click "Revoke" on the old token
3. Click "Create New API Token"
4. Replace the old `kaggle.json` with the new one

## ✅ Verification Checklist

- [ ] Kaggle account created
- [ ] API token downloaded
- [ ] `kaggle.json` placed in correct location
- [ ] File permissions set (Linux/Mac)
- [ ] Dataset terms accepted on Kaggle
- [ ] Setup script runs without errors

## 📞 Need Help?

- Kaggle API Documentation: [https://www.kaggle.com/docs/api](https://www.kaggle.com/docs/api)
- Kaggle Support: [https://www.kaggle.com/support](https://www.kaggle.com/support)

---

Once setup is complete, you can run the scripts to download and work with the GTZAN dataset!

