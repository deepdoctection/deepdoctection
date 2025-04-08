import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;

import java.io.InputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Map;

public class OCRConfig {
    public static Map<String, Object> loadOCRConfig() {
        Yaml yaml = new Yaml();
        try (InputStream inputStream = new FileInputStream(new File("src/main/resources/ocr_config.yml"))) {
            return yaml.load(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
            return Collections.emptyMap();
        }
    }
}
