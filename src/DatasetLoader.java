import weka.attributeSelection.AttributeSelection;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.HashMap;
import java.util.Map;

public class DatasetLoader {
    private HashMap<String, Instances> datasets = new HashMap<String, Instances>();

    public DatasetLoader () {
        load_dataset("CM1", "datasets/promise/cm1.arff");
        load_dataset("JM1", "datasets/promise/jm1.arff");
        load_dataset("KC1", "datasets/promise/kc1.arff");
        load_dataset("KC2", "datasets/promise/kc2.arff");
        load_dataset("PC1", "datasets/promise/pc1.arff");
    }

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

    public Instances get_dataset (String dataset_name) {
        return datasets.get(dataset_name);
    }
    public HashMap<String, Instances> get_datasets() { return datasets; }

    // Preprocess a single dataset (apply ReplaceMissingValues filter)
    private Instances preprocess(Instances data) {
        try {
            ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
            replaceMissing.setInputFormat(data);
            return Filter.useFilter(data, replaceMissing);
        } catch (Exception e) {
            e.printStackTrace();
            return data;  // Return original data if preprocessing fails
        }
    }

    // Preprocess all datasets in the HashMap
    public void preprocess_datasets() {
        for (Map.Entry<String, Instances> entry : datasets.entrySet()) {
            Instances processedData = preprocess(entry.getValue());
            datasets.put(entry.getKey(), processedData);
        }
        System.out.println("All datasets preprocessed successfully!");
    }

    public void set_dataset (String key, Instances dataset) {
        datasets.put(key, dataset);
    }

    public void apply_feature_selection (AttributeSelection selector) throws Exception {
        for (Map.Entry<String, Instances> entry : datasets.entrySet()) {
            System.out.println("Number of attributes before selection: " + entry.getValue().numAttributes());
            // Perform attribute selection on the dataset
            selector.SelectAttributes(entry.getValue());
            // Reduce dimensionality of the dataset
            datasets.put(entry.getKey(), selector.reduceDimensionality(entry.getValue()));
            System.out.println("Number of attributes after selection: " + entry.getValue().numAttributes());
        }
    }
}