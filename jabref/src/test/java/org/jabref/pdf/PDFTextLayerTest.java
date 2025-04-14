import org.junit.Test;
import static org.junit.Assert.*;

public class PDFTextLayerTest {
    @Test
    public void testAddTextLayer() {
        String extractedText = "This is OCR-extracted text.";
        File inputPdf = new File("document.pdf");
        File outputPdf = new File("output.pdf");

        PDFTextLayer.addTextLayer(inputPdf, extractedText, outputPdf);
        assertTrue(outputPdf.exists());
    }
}
