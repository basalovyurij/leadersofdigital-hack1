package org.leadersofdigital.hack1;

import java.io.IOException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

/**
 *
 * @author yurij
 */
public class Main {
    
    private static AddressTransformer transformer;
    
    public static void main(String[] args) throws IOException {
        String file = "../bad.csv";
        transformer = new AddressTransformer();
        
        SparkConf conf = new SparkConf()
                .setMaster("local[4]")
                .setAppName("Address Transformer")
                .set("spark.hadoop.validateOutputSpecs", "false");
        JavaSparkContext sc = new JavaSparkContext(conf);
        
        long t1 = System.currentTimeMillis();
        
        JavaRDD<String> x = sc.textFile(file, 4);        
        x.map(t -> pipeline(t)).saveAsTextFile("../result_spark");
        
        long t2 = System.currentTimeMillis();
        System.out.println(String.format("Time %s ms", t2 - t1));
        
        sc.stop();
    }
    
    private static String pipeline(String line) {
        String[] row = line.split(";");
        String res = transformer.tranfsorm(row[1]);
        return String.join(";", new String[] { row[0], row[1], res });
    }
}
