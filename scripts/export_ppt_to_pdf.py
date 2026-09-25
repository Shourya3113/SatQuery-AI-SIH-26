"""
Convert SIH2026-IDEA-Presentation-Format.pptx to SIH2026-IDEA-Presentation-Format.pdf
using native Microsoft PowerPoint COM automation.
"""

import sys
from pathlib import Path
import win32com.client

ppt_path = Path("SIH2026-IDEA-Presentation-Format.pptx").resolve()
pdf_path = Path("SIH2026-IDEA-Presentation-Format.pdf").resolve()

print(f"Converting {ppt_path} to {pdf_path}...")

powerpoint = win32com.client.Dispatch("PowerPoint.Application")
# Keep window minimized or hidden
deck = powerpoint.Presentations.Open(str(ppt_path), WithWindow=False)

# ppSaveAsPDF format constant is 32
deck.SaveAs(str(pdf_path), 32)
deck.Close()
powerpoint.Quit()

print(f"Successfully generated: {pdf_path} (Size: {pdf_path.stat().st_size} bytes)")
