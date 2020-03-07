Comandi per compilare senza bibitex
latex thesis | dvips -Ppdf -G0 -ta4 thesis.dvi | ps2pdf14 thesis.ps thesis.pdf

Comandi con bibitex
latex thesis | bibtex thesis | latex thesis | latex thesis | dvips -Ppdf -G0 -ta4 thesis.dvi | ps2pdf14 thesis.ps thesis.pdf