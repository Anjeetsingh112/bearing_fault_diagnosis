# Deployment Guide

Step-by-step guide to deploy the SHAP-Enhanced Bearing Fault Diagnosis dashboard live using **Streamlit Community Cloud** (free tier).

---

## Prerequisites

- [ ] GitHub account (https://github.com)
- [ ] Streamlit Community Cloud account (https://streamlit.io/cloud)
- [ ] Git installed locally (`git --version` to check)
- [ ] Trained models in `models/` folder (run `python pipeline.py` if not done)

---

## Step 1: Verify Project Files

Before deploying, ensure these files exist in your project root:

```
project/
  app.py                  # Streamlit entry point (REQUIRED)
  pipeline.py
  requirements.txt        # Dependencies (REQUIRED for deployment)
  .gitignore              # Excludes dataset and cache
  .streamlit/config.toml  # Dark theme config
  README.md
  DEPLOYMENT.md           # This file
  models/                 # Trained models (37 MB total — OK)
    model_DE_12k.pkl
    model_DE_48k.pkl
    model_FE_12k.pkl
    model_FE_48k.pkl
  outputs/shap/           # Pre-generated SHAP plots (for report)
  utils/
    __init__.py
    feature_extraction.py
    model_loader.py
    shap_utils.py
```

**Important:** `CWRU-dataset/` (657 MB) is excluded by `.gitignore` — users upload their own .mat files via the dashboard.

---

## Step 2: Initialize Git Repository

Open a terminal in the project root:

```bash
cd "C:\Users\hp\Desktop\BTP Poject\major project"

git init
git add .
git status
```

Check `git status` output carefully — make sure `CWRU-dataset/` is NOT listed (it should be ignored).

If you see the dataset in the status, run:
```bash
git rm -r --cached CWRU-dataset/
```

Then commit:

```bash
git commit -m "Initial commit: SHAP-Enhanced Bearing Fault Diagnosis System"
```

---

## Step 3: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `bearing-fault-diagnosis` (or any name you prefer)
3. Description: `SHAP-Enhanced Digital Twin for Bearing Fault Diagnosis`
4. Set to **Public** (required for free Streamlit Cloud)
5. **DO NOT** initialize with README, .gitignore, or license (you already have them)
6. Click **Create repository**

---

## Step 4: Push to GitHub

GitHub will show the commands. Run these locally:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/bearing-fault-diagnosis.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

**Authentication:** If prompted, use a Personal Access Token instead of your password:
- GitHub → Settings → Developer settings → Personal access tokens → Generate new token (classic)
- Select scope: `repo`
- Copy the token and use it as the password

**If push fails due to file size:**
Individual files must be < 100 MB for GitHub. Your models are 5-14 MB each, so this should not happen. If it does, check which file is too large:
```bash
find . -size +50M -type f
```

---

## Step 5: Deploy to Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Click **Sign in with GitHub** (authorize Streamlit to access your repos)
3. Click **New app**
4. Fill in the form:
   - **Repository:** `YOUR_USERNAME/bearing-fault-diagnosis`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** choose a subdomain like `bearing-diagnosis` → `bearing-diagnosis.streamlit.app`
5. Click **Advanced settings** (optional):
   - **Python version:** 3.11 (recommended)
6. Click **Deploy**

Streamlit will now:
- Clone your repo
- Install dependencies from `requirements.txt` (3-5 minutes)
- Launch your app

---

## Step 6: Wait for Build

Watch the logs on the Streamlit deployment page. You'll see:

```
[install] Installing dependencies from requirements.txt...
[install] Successfully installed streamlit plotly shap xgboost ...
[run] Starting Streamlit app...
You can now view your Streamlit app in your browser.
```

Build takes **3-5 minutes** on first deploy. Subsequent deploys are faster (~1 minute).

---

## Step 7: Test Your Live App

Once live, your app is available at:
```
https://bearing-diagnosis.streamlit.app
```

Test these scenarios:
1. **Upload test**: Download a sample `.mat` file from the CWRU dataset (or provide a link in your README) and upload it
2. **Model switching**: Verify all 4 models (DE_12k, DE_48k, FE_12k, FE_48k) load correctly
3. **SHAP computation**: Check SHAP tab works for all models
4. **Alert system**: Upload a fault file and verify the alert appears

---

## Step 8: Update README with Live Link

Add a badge and live link to your `README.md`:

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bearing-diagnosis.streamlit.app)

**Live Demo:** https://bearing-diagnosis.streamlit.app
```

Commit and push:
```bash
git add README.md
git commit -m "Add live demo link"
git push
```

Streamlit Cloud **automatically redeploys** on every push to `main`.

---

## Common Issues & Fixes

### Issue 1: "Module not found" error
**Cause:** Missing dependency in `requirements.txt`

**Fix:** Add the missing package and push:
```bash
echo "package-name>=version" >> requirements.txt
git add requirements.txt
git commit -m "Add missing dependency"
git push
```

### Issue 2: Model file too large
**Cause:** Single file > 100 MB (GitHub limit)

**Fix:** Either compress the model or use Git LFS:
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add models/
git commit -m "Use Git LFS for models"
git push
```

### Issue 3: App runs out of memory
**Cause:** Streamlit Cloud free tier has 1 GB RAM

**Fix:** Reduce memory usage:
- Use `@st.cache_resource` for model loading (already done in `utils/model_loader.py`)
- Reduce SHAP sample size for visualization
- Only load the selected model (not all 4 at once) — already implemented

### Issue 4: SHAP computation is slow
**Cause:** XGBoost TreeExplainer on 500-tree model takes time

**Fix:** This is expected. First SHAP call takes 2-5 seconds; subsequent calls are cached.

### Issue 5: CWRU dataset accidentally pushed
**Fix:**
```bash
git rm -r --cached CWRU-dataset/
git commit -m "Remove dataset from tracking"
git push
```

---

## Step 9 (Optional): Custom Domain

Streamlit Cloud free tier uses `*.streamlit.app` subdomains. For a custom domain:

1. Upgrade to Streamlit Teams (paid) OR
2. Use Cloudflare Workers as a proxy (free)
3. Or deploy to your own server (see alternatives below)

---

## Alternative Deployment Options

### Option A: Render.com (Free)
- Create account at https://render.com
- New Web Service → Connect GitHub repo
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- Free tier sleeps after 15 min of inactivity

### Option B: Hugging Face Spaces (Free)
- Go to https://huggingface.co/new-space
- Select Streamlit SDK
- Upload your files or connect to GitHub
- Always-on free tier

### Option C: Railway.app (Free credits)
- Connect GitHub at https://railway.app
- Detects Streamlit automatically
- $5 free credit per month

### Option D: AWS / GCP / Azure (Production)
- For production deployment at scale
- Requires Docker + paid infrastructure
- See `Dockerfile` (not included, ask if needed)

---

## Maintenance

### Update app after code changes
```bash
git add .
git commit -m "Describe your changes"
git push
```
Streamlit Cloud auto-redeploys in ~1 minute.

### Update models
```bash
python pipeline.py             # Retrain locally
git add models/
git commit -m "Update trained models"
git push
```

### Monitor app
- View logs: Streamlit Cloud → Your App → Manage app → Logs
- Check usage: Streamlit Cloud → Analytics
- Restart app: Streamlit Cloud → Settings → Reboot

---

## Quick Reference — All Commands

```bash
# One-time setup
cd "C:\Users\hp\Desktop\BTP Poject\major project"
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/bearing-fault-diagnosis.git
git push -u origin main

# Deploy at: https://share.streamlit.io

# Update workflow
git add .
git commit -m "Update: description of change"
git push
```

---

## Summary

| Step | Time | What Happens |
|------|------|-------------|
| 1. Verify files | 1 min | Check project structure |
| 2. Git init | 2 min | Create local repo |
| 3. GitHub repo | 2 min | Create remote repo |
| 4. Push | 3 min | Upload to GitHub (37 MB models) |
| 5. Streamlit Cloud | 2 min | Connect + configure |
| 6. Build | 5 min | Install dependencies + start app |
| 7. Test | 5 min | Verify all features work |
| **Total** | **~20 min** | **Live application** |

Your app will be publicly accessible at: `https://YOUR_APP_NAME.streamlit.app`
