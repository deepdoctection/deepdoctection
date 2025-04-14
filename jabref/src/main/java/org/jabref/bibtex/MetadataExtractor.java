import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;

public class MetadataExtractor {
    public static String extractMetadata(String extractedText) {
        try {
            URL url = new URL("http://localhost:8080/api/metadata");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);
            connection.getOutputStream().write(extractedText.getBytes());

            return "Extracted Metadata";
        } catch (IOException e) {
            e.printStackTrace();
            return "";
        }
    }
}
