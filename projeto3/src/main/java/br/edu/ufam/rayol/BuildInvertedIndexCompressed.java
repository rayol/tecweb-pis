package br.edu.ufam.rayol;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MapFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import tl.lin.data.array.ArrayListWritable;
import tl.lin.data.fd.Object2IntFrequencyDistribution;
import tl.lin.data.fd.Object2IntFrequencyDistributionEntry;
import tl.lin.data.pair.PairOfInts;
import tl.lin.data.pair.PairOfObjectInt;
import tl.lin.data.pair.PairOfStringInt;
import tl.lin.data.pair.PairOfWritables;

public class BuildInvertedIndexCompressed extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(BuildInvertedIndexCompressed.class);

    private static class MyMapper extends Mapper<LongWritable, Text, PairOfStringInt, IntWritable> {
        private static final Text WORD = new Text();
        private static final Object2IntFrequencyDistribution<String> COUNTS =
                new Object2IntFrequencyDistributionEntry<String>();

        @Override
        public void map(LongWritable docno, Text doc, Context context)
                throws IOException, InterruptedException {
            String text = doc.toString();

            // Tokeniza cada linha
            List<String> tokens = new ArrayList<String>();
            StringTokenizer itr = new StringTokenizer(text);
            while (itr.hasMoreTokens()) {
                String w = itr.nextToken().toLowerCase().replaceAll("(^[^a-z]+|[^a-z]+$)", "");
                if (w.length() == 0) continue;
                tokens.add(w);
            }

            // Cria um histograma dos termos
            COUNTS.clear();
            for (String token : tokens) {
                COUNTS.increment(token);
            }

            // Emite os postings
            for (PairOfObjectInt<String> e : COUNTS) {
                WORD.set(e.getLeftElement());
                context.write(new PairOfStringInt(WORD.toString(), (int) docno.get()) , new IntWritable(e.getRightElement()));
            }

        }
    }

	//reducer baseado no apresentado pelo jim li
    private static class MyReducer extends
            Reducer<PairOfStringInt, IntWritable, Text, PairOfWritables<VIntWritable, BytesWritable>> {

        private String PrevTerm = null;
        private final static ByteArrayOutputStream Postings = new ByteArrayOutputStream();
        private final static DataOutputStream dataOfPostings = new DataOutputStream(Postings);
        private int docFrequency = 0;
        private int termFrequency = 0;
        private int prevDocNo = 0;

        @Override
        public void reduce(PairOfStringInt key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            Iterator<IntWritable> iter = values.iterator();

            if((!key.getLeftElement().equals(PrevTerm)) && (PrevTerm != null)) {
                context.write(new Text(PrevTerm), new PairOfWritables<VIntWritable, BytesWritable>(new VIntWritable(docFrequency), new BytesWritable(Postings.toByteArray())));
                docFrequency = 0;
                prevDocNo = 0;
                Postings.reset();
            }

            PrevTerm = key.getLeftElement();
            while(iter.hasNext()) {
                docFrequency++;
                termFrequency = iter.next().get();
                WritableUtils.writeVInt(dataOfPostings, key.getRightElement() - prevDocNo);
                WritableUtils.writeVInt(dataOfPostings, termFrequency);

                prevDocNo = key.getRightElement();
            }
            
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {

            context.write(new Text(PrevTerm), new PairOfWritables<VIntWritable, BytesWritable>(new VIntWritable(docFrequency), new BytesWritable(Postings.toByteArray())));
            Postings.reset();
            super.cleanup(context);
        }
    }

    protected static class MyPartitioner extends Partitioner<PairOfStringInt, IntWritable> {
        @Override
        public int getPartition(PairOfStringInt key, IntWritable value, int numReduceTasks) {
            return (key.getLeftElement().hashCode() & Integer.MAX_VALUE)  % numReduceTasks;
        }
    }


    private BuildInvertedIndexCompressed() {}

    public static class Args {
        @Option(name = "-input", metaVar = "[path]", required = true, usage = "input path")
        public String input;

        @Option(name = "-output", metaVar = "[path]", required = true, usage = "output path")
        public String output;

        @Option(name = "-reducers", metaVar = "[num]", required = false, usage = "number of reducers")
        public int numReducers = 1;
    }

    /**
     * Runs this tool.
     */
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

        LOG.info("Tool: " + BuildInvertedIndexCompressed.class.getSimpleName());
        LOG.info(" - input path: " + args.input);
        LOG.info(" - output path: " + args.output);

        Job job = Job.getInstance(getConf());
        job.setJobName(BuildInvertedIndexCompressed.class.getSimpleName());
        job.setJarByClass(BuildInvertedIndexCompressed.class);

        job.setNumReduceTasks(args.numReducers);

        FileInputFormat.setInputPaths(job, new Path(args.input));
        FileOutputFormat.setOutputPath(job, new Path(args.output));

        job.setMapOutputKeyClass(PairOfStringInt.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(PairOfWritables.class);
        job.setOutputFormatClass(MapFileOutputFormat.class);

        job.setMapperClass(MyMapper.class);
        job.setReducerClass(MyReducer.class);
        job.setPartitionerClass(MyPartitioner.class);

        // Delete the output directory if it exists already.
        Path outputDir = new Path(args.output);
        FileSystem.get(getConf()).delete(outputDir, true);

        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        System.out.println("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return 0;
    }

    /**
     * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
     */
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new BuildInvertedIndexCompressed(), args);
    }
}
