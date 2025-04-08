import java.io.File;

public interface OCRInterface {
    String extractTextFromImage(File pdfFile) throws OCRException;
    String getEngineName();
}

class OCRException extends Exception {
    public OCRException(String message) {
        super(message);
    }

    public OCRException(String message, Throwable cause) {
        super(message, cause);
    }
}
