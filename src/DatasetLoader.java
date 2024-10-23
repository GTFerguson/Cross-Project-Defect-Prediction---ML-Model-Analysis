import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import java.util.HashMap;

public class DatasetLoader {
    private HashMap<String, Instances> datasets = new HashMap<String, Instances>();

    public void load_dataset (String dataset_name, String filepath) {
        try {
            DataSource source = new DataSource(filepath);
            Instances dataset = source.getDataSet();

            // Set the class index attribute if it is not provided
            if (dataset.classIndex() == -1)
                dataset.setClassIndex(dataset.numAttributes()- 1);

            datasets.put(dataset_name, dataset);
            System.out.println(dataset_name + " successfully loaded!");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void load_nasa_datasets () {
        load_dataset("CM1", "datasets/promise/cm1.arff");
        load_dataset("JM1", "datasets/promise/jm1.arff");
        load_dataset("KC1", "datasets/promise/kc1.arff");
        load_dataset("KC2", "datasets/promise/kc2.arff");
        load_dataset("PC1", "datasets/promise/pc1.arff");
    }

    public Instances get_dataset (String dataset_name) {
        return datasets.get(dataset_name);
    }
    public HashMap<String, Instances> get_datasets() { return datasets; }
}