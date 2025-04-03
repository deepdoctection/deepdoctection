import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.Tesseract;
import java.io.File;
import java.io.FileInputStream;

public class TesseractOCR implements OCRInterface {
    @Override
    public String extractTextFromImage(File pdfFile) {
        try {
            FileInputStream inputStream = new FileInputStream(pdfFile);
            ITesseract instance = new Tesseract();
            instance.setLanguage("eng");
            return instance.doOCR(inputStream);
        } catch (Exception e) {
            e.printStackTrace();
            return "";
        }
    }
}
