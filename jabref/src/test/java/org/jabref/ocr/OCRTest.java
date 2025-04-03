import org.junit.Test;
import static org.junit.Assert.*;

public class OCRTest {
    @Test
    public void testTesseractOCR() {
        OCRInterface tesseractOCR = new TesseractOCR();
        String extractedText = tesseractOCR.extractTextFromImage(new File("document.pdf"));
        assertTrue(extractedText.length() > 0);
    }

    @Test
    public void testDoctrOCR() {
        OCRInterface doctrOCR = new DoctrOCR();
        String extractedText = doctrOCR.extractTextFromImage(new File("document.pdf"));
        assertTrue(extractedText.length() > 0);
    }
}
