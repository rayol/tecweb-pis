package br.edu.ufam.rayol;

import com.google.common.collect.Sets;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;
import org.apache.hadoop.io.LongWritable;
import tl.lin.data.pair.PairOfStrings;
import java.io.IOException;
import java.util.*;


public class PairsPMI extends Configured implements Tool {
    
	private static final Logger LOG = Logger.getLogger(PairsPMI.class);


//jOB 1 - Mapper do WordCounter
	
    protected static class WordCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
	private final static Text KEY = new Text();
	private final static IntWritable ONE = new IntWritable(1);

    @Override
    public void map(LongWritable key, Text value, Context context)
        throws IOException, InterruptedException {
        
      String line = ((Text) value).toString();
      StringTokenizer itr = new StringTokenizer(line);
      
      while (itr.hasMoreTokens()) {
        String w = itr.nextToken().toLowerCase().replaceAll("(^[^a-z]+|[^a-z]+$)", "");
        
        
        if (w.length() == 0) continue;
        
        KEY.set(w);
        context.write(KEY, ONE);
      }
    }
  }
 
  //JOB 1 - InMapper Combiner WordCount
    protected static class WordCountMapperIMC extends Mapper<LongWritable, Text, Text, IntWritable> {
    
    private final HashMap<String, Integer> counts = new HashMap<String, Integer>();

    @Override
    public void map(LongWritable key, Text value, Context context)
        throws IOException, InterruptedException {
        
      String line = ((Text) value).toString();
      StringTokenizer itr = new StringTokenizer(line);
      
      while (itr.hasMoreTokens()) {
        String word = itr.nextToken().toLowerCase().replaceAll("(^[^a-z]+|[^a-z]+$)", "");
        
        if (word.length() == 0) continue;

        if (counts.containsKey(word)) {
          counts.put(word, counts.get(word)+1);
        } else {
          counts.put(word, 1);
        }
      }
    }
    
    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
      IntWritable cnt = new IntWritable();
      Text token = new Text();

      for (Map.Entry<String, Integer> entry : counts.entrySet()) {
        token.set(entry.getKey());
        cnt.set(entry.getValue());
        context.write(token, cnt);
      }
    }
  }
 
  
//JOB 1 - Reducer WordCount
    protected static class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private final static IntWritable SUM = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {
            

            int sum = 0;
            for(IntWritable value : values){
        		sum += value.get();
			}
            SUM.set(sum);
            context.write(key, SUM);
        }
    }

    
/// JOB 2 - Mapper Coocorrencia
    protected static class PairMapper extends Mapper<LongWritable, Text, PairOfStrings, IntWritable> {
        private final static IntWritable ONE = new IntWritable(1);
        private final static PairOfStrings PAIR = new PairOfStrings();

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException{

            String line = ((Text) value).toString();
            StringTokenizer itr = new StringTokenizer(line);

            Set<String> sortedTerms = new TreeSet<String>();
            
            while(itr.hasMoreTokens()){
              String word = itr.nextToken().toLowerCase().replaceAll("(^[^a-z]+|[^a-z]+$)", "");
              
              if (word.length() == 0) continue;
              
              sortedTerms.add(word);
            }

            String left = "";
            String right = "";

            String[] terms = new String[sortedTerms.size()]; 
            sortedTerms.toArray(terms);

            for(int leftTermIndex = 0; leftTermIndex < terms.length; leftTermIndex++){
              for(int rightTermIndex = leftTermIndex + 1; rightTermIndex < terms.length; rightTermIndex++) {
                left = terms[leftTermIndex];
                right = terms[rightTermIndex];

                PAIR.set(left, right);
                context.write(PAIR, ONE);

              }
            }
            
        }
    }


    protected static class PairCombiner extends Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
        private static final IntWritable SUM = new IntWritable();

        @Override
        public void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            Iterator<IntWritable> iter = values.iterator();
            while (iter.hasNext()) {
                sum += iter.next().get();
            }

            SUM.set(sum);
            context.write(key, SUM);
        }
    }



    private static class PairReducer extends Reducer<PairOfStrings, IntWritable, PairOfStrings, DoubleWritable> {
        

    	private static Map<String, Integer> termTotals = new HashMap<String, Integer>();
        private static DoubleWritable PMI = new DoubleWritable();
        private static double totalDocs = 0.0;


        @Override
        public void setup(Context context) throws IOException {
            Path filePath = new Path("tmp/part-r-00000");
            FileSystem fs = FileSystem.get(context.getConfiguration());
            Text key = new Text();
            IntWritable value = new IntWritable();
            SequenceFile.Reader reader = new SequenceFile.Reader(context.getConfiguration(), SequenceFile.Reader.file(filePath));
            while (reader.next(key, value)) {
                
            	int val = Integer.parseInt(value.toString());
            	totalDocs += val;
            	
            	termTotals.put(key.toString(), val);

            }
            reader.close();
        } 
        

        @Override
        public void reduce(PairOfStrings pair, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
        	 
            // Somente calcula o PMI se o par ocorrer mais de 10 vezes
          
          int pairSum = 0;
          for(IntWritable value : values) {
            pairSum += value.get();
          }
          
          if(pairSum >= 10){

            
            String left = pair.getLeftElement();
            String right = pair.getRightElement();

            
            double probPair = pairSum / totalDocs;
            double probLeft = termTotals.get(left) / totalDocs;
            double probRight = termTotals.get(right) / totalDocs;

            double pmi = Math.log(probPair / (probLeft * probRight));


            pair.set(left, right);

            PMI.set(pmi);
            context.write(pair, PMI);
          }

        }

      }




    public PairsPMI() {}

    public static class Args {
        @Option(name = "-input", metaVar = "[path]", required = true, usage = "input path")
        public String input;

        @Option(name = "-output", metaVar = "[path]", required = true, usage = "output path")
        public String output;

        @Option(name = "-reducers", metaVar = "[num]", required = false, usage = "number of reducers")
        public int numReducers = 1;

        @Option(name = "-imc", usage = "use in-mapper combining")
        boolean imc = false;
    }

    public int run(String[] argv) throws Exception {
        Args args = new Args();
        CmdLineParser parser = new CmdLineParser(args, ParserProperties.defaults().withUsageWidth(100));

        try {
            parser.parseArgument(argv);
        } catch (CmdLineException e) {
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
            return -1;
        }

        LOG.info("Tool: " + PairsPMI.class.getSimpleName());
        LOG.info(" - input path: " + args.input);
        LOG.info(" - output path: " + args.output);
        LOG.info(" - number of reducers: " + args.numReducers);
        LOG.info(" - use in-mapper combining: " + args.imc);

        Configuration conf = getConf();


        Job job = Job.getInstance(conf);
        job.setJobName(PairsPMI.class.getSimpleName());
        job.setJarByClass(PairsPMI.class);

        job.setNumReduceTasks(1);

        FileInputFormat.setInputPaths(job, new Path(args.input));
        FileOutputFormat.setOutputPath(job, new Path("tmp")); 

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        job.setMapperClass(args.imc ? WordCountMapper.class : WordCountMapperIMC.class);

        job.setReducerClass(WordCountReducer.class);



        Path outputDir = new Path("tmp");

        FileSystem.get(conf).delete(outputDir, true);


        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");


        Job job2 = Job.getInstance(conf);
        job2.setJobName(PairsPMI.class.getSimpleName());
        job2.setJarByClass(PairsPMI.class);

        job2.setNumReduceTasks(args.numReducers);

        FileInputFormat.setInputPaths(job2, new Path(args.input));
        FileOutputFormat.setOutputPath(job2, new Path(args.output));

        job2.getConfiguration().setInt("mapred.max.split.size", 1024 * 1024 * 64);
        job2.getConfiguration().set("mapreduce.map.memory.mb", "3072");
        job2.getConfiguration().set("mapreduce.map.java.opts", "-Xmx3072m");
        job2.getConfiguration().set("mapreduce.reduce.memory.mb", "3072");
        job2.getConfiguration().set("mapreduce.reduce.java.opts", "-Xmx3072m");

        Path outputDir2 = new Path(args.output);
        FileSystem.get(conf).delete(outputDir2, true);

        job2.setMapOutputKeyClass(PairOfStrings.class);
        job2.setMapOutputValueClass(IntWritable.class);
        job2.setOutputKeyClass(PairOfStrings.class);
        job2.setOutputValueClass(DoubleWritable.class);
        job2.setOutputFormatClass(TextOutputFormat.class);

        job2.setMapperClass(PairMapper.class);
        job2.setCombinerClass(PairCombiner.class);
        job2.setReducerClass(PairReducer.class);

        job2.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");


        return 0;
    }

    //
    // Dispatches command-line arguments to the tool via the {@code ToolRunner}.
    // /
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new PairsPMI(), args);
    }
}
