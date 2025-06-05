# 📊 Repository Integration: Add APA Writing to Existing Research Projects

**Problem**: You have an existing research repository (e.g., "lead-measure") with data, analysis, and findings. Now you want to turn it into a publication-quality APA paper.

**Solution**: Integrate this AI Academic Writing Template into your existing project.

---

## 🎯 **Method 1: Template Subfolder (Recommended)**

### **Step 1: Add Template to Your Project**
```bash
# In your existing repository
cd /path/to/your-research-project

# Add the template as a subfolder
git submodule add https://github.com/yourusername/ai-academic-writer.git paper
# OR simply copy the template:
git clone https://github.com/yourusername/ai-academic-writer.git paper
```

### **Step 2: Your Project Structure**
```
your-research-project/
├── data/                    # Your existing data
├── analysis/               # Your existing analysis  
├── results/                # Your existing results
├── README.md              # Your existing README
└── paper/                 # ← NEW: APA writing template
    ├── academic_paper.qmd
    ├── references.bib
    ├── _extensions/
    └── README.md
```

### **Step 3: Customize for Your Research**
```bash
cd paper
# Edit the paper template with your research
code academic_paper.qmd  # or any editor
```

Replace template content with your research:
- **Title**: Your research title
- **Authors**: Your team
- **Abstract**: Your study summary  
- **Introduction**: Your literature review
- **Method**: Your methodology
- **Results**: Your findings (reference ../results/ if needed)
- **Discussion**: Your interpretation

### **Step 4: Reference Your Existing Work**
In the paper, reference your existing analysis:

```markdown
# Results

Our analysis (see `../analysis/main_analysis.R`) revealed significant effects...

The model performance metrics are detailed in `../results/model_performance.csv`.
```

Or include R code that reads your existing results:
```{r}
# Load results from your existing analysis
results <- read.csv("../results/model_performance.csv")
knitr::kable(results, caption = "Model Performance Results")
```

---

## 🎯 **Method 2: Copy Template Files**

### **Step 1: Copy Essential Files**
```bash
# In your existing repository
mkdir paper-draft
cd paper-draft

# Copy only what you need:
cp /path/to/ai-academic-writer/academic_paper.qmd ./manuscript.qmd
cp /path/to/ai-academic-writer/references.bib ./
cp -r /path/to/ai-academic-writer/_extensions ./
```

### **Step 2: Minimal Integration**
Your structure:
```
your-research-project/
├── data/
├── analysis/  
├── results/
├── paper-draft/
│   ├── manuscript.qmd      # Your paper
│   ├── references.bib      # Your citations
│   └── _extensions/        # APA formatting
└── README.md
```

---

## 🤖 **Method 3: AI-Assisted Content Transfer**

### **For Your "lead-measure" Leadership Embeddings Project**

1. **Add template to your project**:
```bash
cd lead-measure
git clone https://github.com/yourusername/ai-academic-writer.git paper
cd paper
```

2. **Create a project context file** (`RESEARCH_CONTEXT.md`):
```markdown
# Lead-Measure Project Context

## Research Summary
- **Topic**: Leadership measurement using embeddings
- **Data**: [Describe your datasets]
- **Methods**: [Describe your analysis approach]
- **Key Findings**: [Summarize main results]

## Existing Analysis Files
- `analysis/embedding_analysis.py` - Main analysis script
- `results/leadership_scores.csv` - Computed leadership scores  
- `results/model_validation.png` - Model performance plots
- `data/survey_responses.csv` - Raw survey data

## Target Publication
- **Journal**: [Target journal]
- **Focus**: Novel approach to leadership measurement
- **Key Innovation**: Using embeddings for leadership assessment
```

3. **Give this prompt to a fresh AI in the paper folder**:
```
I have a research project on leadership measurement using embeddings. I want to convert my existing analysis into an APA research paper using this template.

Project context is in RESEARCH_CONTEXT.md. 

Please help me:
1. Review my existing analysis files  
2. Structure this as an empirical research paper
3. Convert my findings into proper APA format
4. Create appropriate tables and figures
5. Write a compelling introduction and discussion

My goal is a publication-ready manuscript that showcases the embedding approach to leadership measurement.
```

---

## 📋 **Template for Any Research Project**

### **Universal Integration Steps**

1. **Assess your project**:
   - What type of paper? (Empirical, theoretical, review)
   - What's your main contribution?
   - What evidence do you have?

2. **Choose integration method**:
   - Subfolder for large projects
   - Copy files for simple papers
   - AI assistance for complex conversion

3. **Customize the template**:
   - Update YAML header with your info
   - Replace sections with your content
   - Add your references to `.bib` file
   - Include your data/results as needed

4. **Compile and iterate**:
   ```bash
   quarto render academic_paper.qmd --to apaquarto-docx
   ```

5. **Refine until perfect**:
   - Check APA formatting
   - Verify citations  
   - Review tables/figures
   - Get feedback from co-authors

---

## 🎯 **Project-Specific Examples**

### **For Data Science Projects**
```{r}
# Reference your existing analysis
source("../analysis/main_analysis.R")

# Include your results
results_summary <- read.csv("../results/summary_stats.csv")
knitr::kable(results_summary, caption = "Analysis Results")

# Include your plots  
knitr::include_graphics("../results/main_findings.png")
```

### **For Experimental Studies**
```markdown
# Method

## Participants
We recruited N participants (see `../data/demographics.csv` for details).

## Procedure  
The experimental protocol is documented in `../methods/protocol.md`.

# Results
Our statistical analysis (`../analysis/stats.R`) revealed...
```

### **For Literature Reviews**
```markdown
# Introduction
We systematically reviewed [topic] (search strategy in `../methods/search_strategy.md`).

Our analysis of 47 papers (bibliographic data: `../data/papers.csv`) revealed...
```

---

## 🎊 **Success Criteria**

Your integration is successful when:

✅ **You can compile** your manuscript with one command  
✅ **APA formatting** is perfect throughout  
✅ **Your existing work** is properly incorporated  
✅ **Citations** resolve correctly  
✅ **Tables/figures** appear professionally  
✅ **Co-authors** can easily collaborate  

---

## 💡 **Pro Tips for Integration**

1. **Keep your existing workflow**: Don't change how you do analysis, just add the paper compilation step

2. **Use relative paths**: Reference your existing files with `../` so the paper folder is portable

3. **Document everything**: Update your main README to mention the paper folder

4. **Version control both**: Commit both your analysis changes AND paper changes

5. **Separate concerns**: Keep data/analysis in main repo, paper writing in subfolder

6. **AI assistance**: The template's Cursor rules will help an AI understand your project structure

**The goal**: Transform any research repository into a publication pipeline with minimal disruption to existing workflow!**