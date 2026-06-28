# Google Form Field Specification for CVision

## Overview

Design your Google Form with **exactly these question titles**. The column headers
in the linked Google Sheet are auto-generated from question text, and the ingestion
module maps them by exact name. If you rename a question, update `COLUMN_MAP` in
`modules/ingestion.py` to match.

---

## Recommended Form Structure

### Section 1 — Personal Information

| # | Question Title (EXACT) | Type | Required |
|---|---|---|---|
| 1 | `Full Name` | Short answer | ✅ Yes |
| 2 | `Email Address` | Short answer | ✅ Yes |
| 3 | `Phone Number` | Short answer | ❌ No |

### Section 2 — Academic Background

| # | Question Title (EXACT) | Type | Required | Notes |
|---|---|---|---|---|
| 4 | `CGPA / GPA` | Short answer | ✅ Yes | Add description: *"Enter on a 4.0 scale, e.g. 3.5"* |
| 5 | `Highest Degree Earned` | Multiple choice | ✅ Yes | Options below ↓ |
| 6 | `Major / Field of Study` | Short answer | ✅ Yes | |

**Choices for question 5 — Highest Degree Earned:**
- `High School Diploma`
- `Bachelor's Degree`
- `Master's Degree`
- `PhD / Doctorate`
- `Other`

### Section 3 — Professional Background

| # | Question Title (EXACT) | Type | Required | Notes |
|---|---|---|---|---|
| 7 | `Years of Professional Experience` | Short answer | ✅ Yes | Add description: *"Enter as a number, e.g. 3 or 3.5"* |
| 8 | `Current or Last Job Title` | Short answer | ❌ No | |
| 9 | `LinkedIn Profile URL` | Short answer | ❌ No | |

### Section 4 — Application Details

| # | Question Title (EXACT) | Type | Required | Notes |
|---|---|---|---|---|
| 10 | `Position Applying For` | Short answer (or Dropdown) | ✅ Yes | Use Dropdown if multiple roles |
| 11 | `Upload Resume / CV` | File upload | ✅ Yes | See settings below ↓ |
| 12 | `Brief Cover Note` | Paragraph | ❌ No | |

**Settings for question 11 — Upload Resume / CV:**
- Allow only specific file types: ✅ `PDF`
- Maximum number of files: `1`
- Maximum file size: `10 MB`

---

## Google Sheet Column Order

When the form is linked to a Sheet, the headers will appear in this order
(Google always prepends Timestamp as column A):

```
A: Timestamp
B: Full Name
C: Email Address
D: Phone Number
E: CGPA / GPA
F: Highest Degree Earned
G: Major / Field of Study
H: Years of Professional Experience
I: Current or Last Job Title
J: LinkedIn Profile URL
K: Position Applying For
L: Upload Resume / CV
M: Brief Cover Note
```

---

## Service Account Setup (for Google Sheet + Drive access)

### Step 1 — Create a Google Cloud Project
1. Go to https://console.cloud.google.com
2. Create a new project (e.g., "CVision")

### Step 2 — Enable APIs
In **APIs & Services > Library**, enable:
- ✅ Google Sheets API
- ✅ Google Drive API

### Step 3 — Create a Service Account
1. Go to **APIs & Services > Credentials**
2. Click **Create Credentials > Service Account**
3. Name it `cvision-service-account`
4. Skip optional fields, click **Done**
5. Click on the created service account → **Keys** tab → **Add Key > Create new key**
6. Choose **JSON** → Download and save as `credentials.json` in your project folder

### Step 4 — Share Access
You must grant the service account access to both the Sheet and the Drive folder:

1. **Google Sheet:** Click Share → paste the service account email (looks like
   `cvision-service-account@your-project.iam.gserviceaccount.com`) → set to **Viewer**

2. **Google Drive folder** (where form uploads are stored):
   - Open Google Drive → find the folder named after your form
   - Right-click → Share → paste the service account email → set to **Viewer**

### Step 5 — Configure .env
```
GOOGLE_SHEET_ID=1abc...xyz       # From your Sheet URL
GOOGLE_CREDENTIALS_PATH=credentials.json
```

---

## File Upload URL Format

When a candidate uploads a PDF via the form, the Sheet cell will contain a
Google Drive URL in one of these formats:

```
https://drive.google.com/open?id=FILE_ID
https://drive.google.com/file/d/FILE_ID/view?usp=drivesdk
```

The ingestion module automatically parses both formats to extract the `FILE_ID`
and downloads the file using the Drive API.

---

## Knockout Filters Applied

| Field | Condition | Default |
|---|---|---|
| CGPA / GPA | Must be ≥ threshold | 3.0 / 4.0 |
| Years of Professional Experience | Must be ≥ threshold | 3 years |
| Highest Degree Earned | Must be Bachelor's or higher | Bachelor's+ |

Candidates who fail any filter are shown in a separate "Filtered Out" panel
in the dashboard with the specific reason for rejection.
