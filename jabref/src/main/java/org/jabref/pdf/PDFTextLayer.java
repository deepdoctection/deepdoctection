import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.font.PDType1Font;

import java.io.File;
import java.io.IOException;

public class PDFTextLayer {
    public static void addTextLayer(File pdfFile, String extractedText, File outputFile) {
        try {
            PDDocument document = PDDocument.load(pdfFile);
            PDPage page = document.getPage(0);
            PDPageContentStream contentStream = new PDPageContentStream(document, page, true, true);

            contentStream.beginText();
            contentStream.setFont(PDType1Font.HELVETICA_BOLD, 12);
            contentStream.newLineAtOffset(50, 750);
            contentStream.showText(extractedText);
            contentStream.endText();
            contentStream.close();

            document.save(outputFile);
            document.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
