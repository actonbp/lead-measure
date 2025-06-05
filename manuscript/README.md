# Leadership Measurement Manuscript (Updated June 4, 2025)

This directory contains the APA-formatted manuscript documenting the leadership measurement analysis findings.

## 📄 Main Manuscript

**`leadership_measurement_paper.qmd`** - The complete manuscript in Quarto format

### Key Findings Documented
- **87.4% accuracy** for IPIP personality constructs (Cohen's d = 1.116)
- **62.9% accuracy** for leadership constructs (Cohen's d = 0.368)
- **24.5 percentage point gap** with high statistical significance (p < 2.22e-16)

## 🚀 Compiling the Manuscript

### Prerequisites
1. **R** (required): https://cran.r-project.org/
2. **Quarto**: https://quarto.org/docs/get-started/
3. **apaquarto extension** (included in `_extensions/`)

### Compilation Command
```bash
# THE CRITICAL COMMAND - Use this exact command for APA formatting
quarto render leadership_measurement_paper.qmd --to apaquarto-docx

# This creates: leadership_measurement_paper.docx
```

## 📊 Manuscript Contents

### Sections Included
1. **Abstract** - Summary of findings with keywords
2. **Introduction** - Construct proliferation problem in leadership research
3. **Method** - Ivan's enhanced contrastive learning approach
4. **Results** - Statistical validation and effect sizes
5. **Discussion** - Implications for leadership theory and measurement
6. **References** - Complete citation list
7. **Tables & Figures** - Including the key t-SNE visualization

### Key Figure
The manuscript includes `top5_coherent_constructs_tsne.png` showing:
- Clear clustering of IPIP personality constructs
- Overlapping clusters for leadership constructs
- Visual evidence of construct redundancy

## ✏️ Updating the Manuscript

### To Add New Content
1. Edit `leadership_measurement_paper.qmd` 
2. Update citations in `references.bib`
3. Re-compile with the command above

### To Update Figures
1. Place new figures in manuscript directory
2. Reference in .qmd file: `![Caption](filename.png)`
3. Figures will be properly formatted at document end

## 📚 References

The `references.bib` file contains all citations including:
- Ivan Hernandez's methodology papers
- Leadership construct validation studies
- Contrastive learning in NLP references
- Psychological measurement theory sources

## 🎯 Manuscript Status

**Current Version**: Complete draft ready for submission
- All results validated with holdout methodology
- Statistical tests properly reported
- APA formatting verified
- Implications clearly articulated

## 📁 Supporting Files

- `CLAUDE.md` - AI assistant instructions for manuscript work
- `_extensions/wjschne/apaquarto/` - Complete APA formatting system
- Previous drafts preserved in git history

## 🚨 Important Notes

1. **Always use exact render command** for proper APA formatting
2. **Check Word output** - Some formatting may need minor tweaks
3. **Verify citations** - Ensure all references are properly linked
4. **Update statistics** - If re-running analysis, update accuracy values

## 🛠️ Troubleshooting

### Common Issues

**Quarto Won't Render**
- Install Quarto from https://quarto.org/docs/get-started/
- Verify `_extensions` folder exists with apaquarto files

**Wrong Formatting**
- Must use `--to apaquarto-docx` flag, not just `--to docx`
- Check that R is installed (required for apaquarto)

**Missing Citations**
- Verify references.bib has correct format
- Ensure citation keys match between .qmd and .bib files