from matplotlib.backends.backend_pdf import PdfPages

class PDFWriter:
    def __init__(self, path):
        self.pdf = PdfPages(path)

    def save_fig(self, fig):
        self.pdf.savefig(fig)
        fig.clf()

    def close(self):
        self.pdf.close()
