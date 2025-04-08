public class DoctrOCR implements OCRInterface {
    @Override
    public String extractTextFromImage(File pdfFile) {
        String command = "python3 doctr_cli.py --pdf " + pdfFile.getPath();
        return "";
    }
}
