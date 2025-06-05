# 🚀 QUICK START: Your First APA Paper in 10 Minutes

**Goal**: Create your own APA research paper from this template in under 10 minutes.

---

## ✅ **STEP 1: Get the Template (2 minutes)**

```bash
# Clone this template to your computer
git clone https://github.com/yourusername/ai-academic-writer.git
cd ai-academic-writer
```

**You now have**: Complete working template with fake research paper

---

## ✅ **STEP 2: Install Software (5 minutes)**

### **Install R**
1. Go to https://cran.r-project.org/
2. Download for your OS (Windows/Mac/Linux) 
3. Install normally

### **Install Quarto**
1. Go to https://quarto.org/docs/get-started/
2. Download for your OS
3. Install normally

### **Test Everything Works**
```bash
# These should both work:
R --version        # Should show R 4.0+
quarto --version   # Should show 1.3+
```

---

## ✅ **STEP 3: Create Your First Paper (2 minutes)**

### **Test the Template**
```bash
# This creates a perfect APA document:
quarto render academic_paper.qmd --to apaquarto-docx

# Open the result:
open academic_paper.docx  # Mac
# or
start academic_paper.docx  # Windows
```

**Expected Result**: Perfect APA paper with fake research about AI academic writing

---

## ✅ **STEP 4: Make It Your Own (1 minute)**

Edit `academic_paper.qmd` and change:

```yaml
title: "YOUR RESEARCH TITLE HERE"
author:
  - name: "YOUR NAME"
    email: "your.email@university.edu"
    corresponding: true
    affiliations:
      - name: "YOUR UNIVERSITY" 
        department: "YOUR DEPARTMENT"
```

**Then recompile:**
```bash
quarto render academic_paper.qmd --to apaquarto-docx
```

---

## 🎯 **COMMON USE CASES**

### **For a Method Section Draft**
1. Keep the YAML header (title, author, etc.)
2. Delete everything except the "# Method" section
3. Replace with your method content
4. Compile to get APA-formatted method section

### **For a Complete Paper**
1. Replace the abstract with your abstract
2. Replace each section with your content:
   - Introduction → Your literature review
   - Method → Your methodology  
   - Results → Your findings
   - Discussion → Your interpretation
3. Update `references.bib` with your citations
4. Compile to get complete APA paper

### **For Collaboration**
1. Share this repository with co-authors
2. Everyone edits `academic_paper.qmd`
3. Use git for version control
4. Compile whenever you want to see APA output

---

## 🔧 **Quick Customization Guide**

### **Change Citations**
Add to `references.bib`:
```bibtex
@article{yourkey2024,
  title={Your Article Title},
  author={Author Name},
  journal={Journal Name}, 
  year={2024}
}
```

Use in text: `[@yourkey2024]`

### **Add Tables**
```markdown
| Column 1 | Column 2 |
|----------|----------|
| Data     | Data     |

: Table Caption
```

### **Add R Code/Figures**
```{r}
#| fig-cap: "Your Figure Caption"

# Your R code here
plot(1:10, 1:10)
```

---

## 🚨 **TROUBLESHOOTING**

### **"quarto: command not found"**
- Install Quarto from https://quarto.org/

### **"R: command not found"**  
- Install R from https://cran.r-project.org/

### **Wrong formatting in Word**
- Make sure you use `--to apaquarto-docx` (NOT just `--to docx`)

### **Tables not at end**
- This is correct! APA format moves tables to end automatically

### **Citations show as [?]**
- Check your `references.bib` file has the correct citation keys

---

## ⚡ **Pro Tips**

1. **Daily workflow**: Edit `.qmd` → Compile → Check Word doc → Repeat
2. **Version control**: Commit changes regularly with git
3. **Collaboration**: Share repository URL, not Word files
4. **AI assistance**: Use Cursor IDE for intelligent writing help
5. **Keep it simple**: Start with basic content, add complexity later

---

**🎊 THAT'S IT! You now have a working APA academic writing system.**

**Next**: Check `README.md` for advanced features and `REPOSITORY_INTEGRATION.md` for using this in existing projects.