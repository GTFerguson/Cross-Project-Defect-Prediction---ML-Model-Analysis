import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import java.util.HashMap;

public class DatasetLoader {
    private HashMap<String, Instances> datasets = new HashMap<String, Instances>();

    public void load_dataset (String dataset_name, String filepath) {
        try {
            DataSource source = new DataSource(filepath);
            Instances dataset = source.getDataSet();
            //dataset.setClassIndex(dataset.numAttributes()- 1);
            datasets.put(dataset_name, dataset);
            System.out.println(dataset_name + " successfully loaded!");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public Instances get_dataset (String dataset_name) {
        return datasets.get(dataset_name);
    }
}
