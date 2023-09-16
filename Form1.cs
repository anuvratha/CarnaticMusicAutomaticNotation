using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

using Microsoft.ML;
using Microsoft.ML.Data;
using NAudio.CoreAudioApi;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using Spectrogram;
using static Tensorflow.Summary.Types;

namespace CarnaticMusicAutomaticNotation
{
    public partial class Form1 : Form
    {
        private string strPath;
        private string strModelPath, strAssetsPath, strTensorFlowModelPath;
        private WasapiCapture AudioDevice;
        private MMDevice[] AudioDevices;
        private double dRecordTimeS;
        private byte[] buffer;
        private Timer timer;
        private int nCount;
        private bool bIsFull = false;
        private int nPos = 0;
        private string strSwara;
        private Dictionary<string, int> dctSwaraToRunning = new Dictionary<string, int>();
        private bool bIsTrainingMode = false;
        private bool bTrainingDone = true;
        private ITransformer model = null;
        private MLContext mlContext;
        private int nAudioDevicesIndex = 0;
        private int nNotationRunning;
        private int fftSize = 16384;
        private int targetWidthPx = 3000;
        private int maxFreq = 3000;
        private int melBins = 300;
        private bool bUseMel = false;
        private double intensity = 5;
        private Colormap colorMap = Colormap.Viridis;
        private bool dB = true;
        private bool bStaticStepSize = true;
        private int nStepSize = 128;
        private bool bSimulateData = false;
        private bool bUseChart = false;

        public Form1()
        {
            strPath = new FileInfo(System.Reflection.Assembly.GetEntryAssembly().Location).DirectoryName + Path.DirectorySeparatorChar;
            strModelPath = strPath + "model" + Path.DirectorySeparatorChar;
            strAssetsPath = strPath + "assets" + Path.DirectorySeparatorChar;
            strTensorFlowModelPath = strPath + "assets" + Path.DirectorySeparatorChar + "inception" + Path.DirectorySeparatorChar + "tensorflow_inception_graph.pb";
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            textBox1.Text = "Anuvratha Narasimhan\r\n";
            textBox1.Text += "PhD Music student\r\n";
            textBox1.Text += "Sri Padmavathi Mahila Viswavidyalayam\r\n";
            textBox1.Text += "anuvrathanarasimhan@gmail.com";

            listBox1.Items.Add("_GAP_");
            listBox1.Items.Add("sa");
            listBox1.Items.Add("ri");
            listBox1.Items.Add("ga");
            listBox1.Items.Add("ma");
            listBox1.Items.Add("pa");
            listBox1.Items.Add("da");
            listBox1.Items.Add("ni");

            AudioDevices = new MMDeviceEnumerator()
                .EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active)
                .ToArray();
            if (AudioDevices.Count() == 0)
            {
                MessageBox.Show("No valid audio devices!");
                Application.Exit();
            }
            mlContext = new MLContext();
            if (File.Exists(strAssetsPath + "model" + Path.DirectorySeparatorChar + "model.zip"))
            {
                model = mlContext.Model.Load(strAssetsPath + "model" + Path.DirectorySeparatorChar + "model.zip", out _);
            }
        }

        private void AudioDevice_DataAvailable(object sender, WaveInEventArgs e)
        {
            for (int i = 0; i < e.BytesRecorded; i++)
            {
                buffer[nPos] = e.Buffer[i];
                nPos = (nPos + 1) % buffer.Length;
                bIsFull |= (nPos == 0);
            }
            if (!bIsFull)
            {
                return;
            }
            nPos = 0;
            bIsFull = false;
            if (bIsTrainingMode)
            {
                var waveFormat = AudioDevice.WaveFormat;
                dctSwaraToRunning.TryGetValue(strSwara, out var nRunning);
                nRunning++;
                dctSwaraToRunning[strSwara] = nRunning;
                using (var writer = new WaveFileWriter(strAssetsPath + "wav" + Path.DirectorySeparatorChar + strSwara + "_" + nRunning + ".wav", waveFormat))
                {
                    writer.Write(buffer, 0, buffer.Length);
                }
                if (bSimulateData)
                {
                    var semitone = (float)Math.Pow(2, 1.0 / 12);
                    var upOneTone = semitone * semitone;
                    var downOneTone = 1.0f / upOneTone;
                    for (int i = 1; i <= 2; i++)
                    {
                        dctSwaraToRunning.TryGetValue(strSwara, out nRunning);
                        nRunning++;
                        dctSwaraToRunning[strSwara] = nRunning;
                        var bufferedWaveProvider = new BufferedWaveProvider(waveFormat);
                        bufferedWaveProvider.ReadFully = false;
                        bufferedWaveProvider.AddSamples(buffer, 0, buffer.Length);
                        var pitchProvider = new SmbPitchShiftingSampleProvider(bufferedWaveProvider.ToSampleProvider());
                        pitchProvider.PitchFactor = i * upOneTone;
                        WaveFileWriter.CreateWaveFile(strAssetsPath + "wav" + Path.DirectorySeparatorChar + strSwara + "_" + nRunning + ".wav", pitchProvider.ToWaveProvider());
                    }
                    for (int i = 1; i <= 2; i++)
                    {
                        dctSwaraToRunning.TryGetValue(strSwara, out nRunning);
                        nRunning++;
                        dctSwaraToRunning[strSwara] = nRunning;
                        var bufferedWaveProvider = new BufferedWaveProvider(waveFormat);
                        bufferedWaveProvider.ReadFully = false;
                        bufferedWaveProvider.AddSamples(buffer, 0, buffer.Length);
                        var pitchProvider = new SmbPitchShiftingSampleProvider(bufferedWaveProvider.ToSampleProvider());
                        pitchProvider.PitchFactor = i * downOneTone;
                        WaveFileWriter.CreateWaveFile(strAssetsPath + "wav" + Path.DirectorySeparatorChar + strSwara + "_" + nRunning + ".wav", pitchProvider.ToWaveProvider());
                    }
                }
            }
            else
            {
                using (var writer = new WaveFileWriter(strAssetsPath + "wav1" + Path.DirectorySeparatorChar + nNotationRunning++ + ".wav", AudioDevice.WaveFormat))
                {
                    writer.Write(buffer, 0, buffer.Length);
                }
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (listBox1.SelectedIndex == -1)
            {
                MessageBox.Show("No swara selected!");
                return;
            }
            if (button1.Text == "Record")
            {
                if (bTrainingDone)
                {
                    RefreshDirs();
                    bTrainingDone = false;
                }
                button1.Text = "Stop";
                bSimulateData = checkBox1.Checked;
                dRecordTimeS = double.Parse(textBox2.Text);
                int nInterval = (int)(dRecordTimeS * 1000);
                timer = new Timer();
                timer.Interval = nInterval;
                timer.Tick += Timer_Tick1;

                AudioDevice = new WasapiCapture(AudioDevices[nAudioDevicesIndex], true, nInterval);
                buffer = new byte[(int)(AudioDevice.WaveFormat.AverageBytesPerSecond * dRecordTimeS)];
                AudioDevice.DataAvailable += AudioDevice_DataAvailable;
                nCount = 0;
                listBox1.Enabled = listBox2.Enabled = false;
                bIsTrainingMode = true;
                timer.Start();
                AudioDevice.StartRecording();
            }
            else
            {
                button1.Text = "Record";
                AudioDevice.StopRecording();
                timer.Stop();
                AudioDevice.DataAvailable -= AudioDevice_DataAvailable;
                timer.Tick -= Timer_Tick1;
                AudioDevice = null;
                timer = null;
                listBox1.Enabled = listBox2.Enabled = true;
                bIsTrainingMode = false;
                if (listBox1.SelectedIndex != -1)
                {
                    listBox1_SelectedIndexChanged(listBox1, null);
                }
            }
        }

        private void Timer_Tick1(object sender, EventArgs e)
        {
            //Console.Beep();
            Tick1();
        }

        private void Timer_Tick2(object sender, EventArgs e)
        {
            //Console.Beep();
            Tick2();
        }

        (double[] audio, int sampleRate) ReadMono(string filePath, double multiplier = 16_000)
        {
            using (var afr = new AudioFileReader(filePath))
            {
                int sampleRate = afr.WaveFormat.SampleRate;
                int bytesPerSample = afr.WaveFormat.BitsPerSample / 8;
                int sampleCount = (int)(afr.Length / bytesPerSample);
                int channelCount = afr.WaveFormat.Channels;
                var audio = new List<double>(sampleCount);
                var buffer = new float[sampleRate * channelCount];
                int samplesRead = 0;
                while ((samplesRead = afr.Read(buffer, 0, buffer.Length)) > 0)
                    audio.AddRange(buffer.Take(samplesRead).Select(x => x * multiplier));
                return (audio.ToArray(), sampleRate);
            }
        }

        private Bitmap GetImage(double[] audio, int sampleRate)
        {
            if (bUseChart)
            {
                var chart = new Chart();
                chart.Series.Add("wave");
                var chartArea = new ChartArea();
                chartArea.AxisX.LabelStyle.Enabled = false;
                chartArea.AxisY.LabelStyle.Enabled = false;
                chartArea.AxisX.MajorGrid.Enabled = false;
                chartArea.AxisY.MajorGrid.Enabled = false;
                chart.ChartAreas.Add(chartArea);
                chart.Series["wave"].ChartType = SeriesChartType.FastLine;
                for (int i = 0; i < audio.Length; i++)
                {
                    chart.Series["wave"].Points.AddXY(i, audio[i]);
                }
                chart.ChartAreas[0].RecalculateAxesScale();
                using (var ms = new MemoryStream())
                {
                    chart.SaveImage(ms, ChartImageFormat.Png);
                    return new Bitmap(ms);
                }
            }
            else
            {
                int stepSize = bStaticStepSize ? nStepSize : audio.Length / targetWidthPx;
                var sg = new SpectrogramGenerator(sampleRate, fftSize: fftSize, stepSize: stepSize, maxFreq: maxFreq);
                sg.Colormap = colorMap;
                sg.Add(audio);
                return bUseMel ? sg.GetBitmapMel(melBins, intensity: intensity, dB: dB) : sg.GetBitmap(intensity: intensity, dB: dB);
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            var sbData1 = new StringBuilder(5000);
            var sbData2 = new StringBuilder(5000);
            foreach (var item in dctSwaraToRunning)
            {
                int nHalf = (int)(item.Value / 2);
                Parallel.For(1, item.Value, i =>
                {
                    var name = item.Key + "_" + i;
                    (double[] audio, int sampleRate) = ReadMono(strAssetsPath + "wav" + Path.DirectorySeparatorChar + name + ".wav");
                    var bmp = GetImage(audio, sampleRate);
                    bmp.Save(strAssetsPath + "images" + Path.DirectorySeparatorChar + name + ".png", ImageFormat.Png);
                    if (i <= nHalf)
                    {
                        lock (sbData1)
                        {
                            sbData1.AppendLine(name + ".png\t" + item.Key);
                        }
                    }
                    else
                    {
                        lock (sbData2)
                        {
                            sbData2.AppendLine(name + ".png\t" + item.Key);
                        }
                    }
                });
            }
            dctSwaraToRunning.Clear();
            File.WriteAllText(strAssetsPath + "images" + Path.DirectorySeparatorChar + "tags.tsv", sbData1.ToString());
            File.WriteAllText(strAssetsPath + "images" + Path.DirectorySeparatorChar + "test-tags.tsv", sbData2.ToString());

            IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: strAssetsPath + "images", inputColumnName: nameof(ImageData.ImagePath))
                // The image transforms transform the images into the model's expected format.
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                .Append(mlContext.Model.LoadTensorFlowModel(strTensorFlowModelPath)
                .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

            IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: strAssetsPath + "images" + Path.DirectorySeparatorChar + "tags.tsv", hasHeader: false);

            model = pipeline.Fit(trainingData);

            IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: strAssetsPath + "images" + Path.DirectorySeparatorChar + "test-tags.tsv", hasHeader: false);
            IDataView predictions = model.Transform(testData);

            IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
            if (imagePredictionData.Count() > 0)
            {
                MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKey", predictedLabelColumnName: "PredictedLabel");
                string strStatus = $"LogLoss is: {metrics.LogLoss}\r\n";
                strStatus += $"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}";
                AppendStatus(strStatus);
                mlContext.Model.Save(model, trainingData.Schema, strAssetsPath + "model" + Path.DirectorySeparatorChar + "model.zip");
            }
            bTrainingDone = true;
        }

        private void RefreshDirs()
        {
            if (Directory.Exists(strAssetsPath + "wav"))
            {
                Directory.Delete(strAssetsPath + "wav", true);
            }
            if (Directory.Exists(strAssetsPath + "images"))
            {
                Directory.Delete(strAssetsPath + "images", true);
            }
            Directory.CreateDirectory(strAssetsPath + "wav");
            Directory.CreateDirectory(strAssetsPath + "images");
        }

        private void RefreshWav1Dir()
        {
            if (Directory.Exists(strAssetsPath + "wav1"))
            {
                Directory.Delete(strAssetsPath + "wav1", true);
            }
            Directory.CreateDirectory(strAssetsPath + "wav1");
        }

        private void listBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            strSwara = (string)listBox1.Items[listBox1.SelectedIndex];
            listBox2.Items.Clear();
            if (dctSwaraToRunning.TryGetValue(strSwara, out var nRunning))
            {
                for (int i = 1; i <= nRunning; i++)
                {
                    listBox2.Items.Add(strSwara + "_" + i);
                }
            }
        }

        private void listBox2_SelectedIndexChanged(object sender, EventArgs e)
        {
            var wavFileName = (string)listBox2.Items[listBox2.SelectedIndex];
            var filePath = strAssetsPath + "wav" + Path.DirectorySeparatorChar + wavFileName + ".wav";
            if (File.Exists(filePath))
            {
                var wave = new AudioFileReader(filePath);
                var outputSound = new WaveOut();
                outputSound.Init(new WaveChannel32(wave));
                outputSound.Play();
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            if (button3.Text == "Record")
            {
                if (model == null)
                {
                    MessageBox.Show("Model doesn't exist. Please train first!");
                    return;
                }
                button3.Text = "Stop";
                dRecordTimeS = double.Parse(textBox2.Text);
                int nInterval = (int)(dRecordTimeS * 1000);
                timer = new Timer();
                timer.Interval = nInterval;
                timer.Tick += Timer_Tick2;

                AudioDevice = new WasapiCapture(AudioDevices[nAudioDevicesIndex], true, nInterval);
                buffer = new byte[(int)(AudioDevice.WaveFormat.AverageBytesPerSecond * dRecordTimeS)];
                AudioDevice.DataAvailable += AudioDevice_DataAvailable;
                nCount = 0;
                nNotationRunning = 1;
                RefreshWav1Dir();
                timer.Start();
                AudioDevice.StartRecording();
            }
            else
            {
                button3.Text = "Record";
                AudioDevice.StopRecording();
                timer.Stop();
                AudioDevice.DataAvailable -= AudioDevice_DataAvailable;
                timer.Tick -= Timer_Tick2;
                AudioDevice = null;
                timer = null;
                Notate();
            }
        }

        private void Notate()
        {
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var arrNotations = new string[nNotationRunning];
            Parallel.For(1, nNotationRunning, i =>
            {
                var filePath = strAssetsPath + "wav1" + Path.DirectorySeparatorChar + i + ".wav";
                if (!File.Exists(filePath))
                {
                    arrNotations[i] = "";
                    return;
                }
                (double[] audio, int sampleRate) = ReadMono(filePath);
                var bmp = GetImage(audio, sampleRate);
                bmp.Save(strAssetsPath + "wav1" + Path.DirectorySeparatorChar + i + ".png", ImageFormat.Png);
                var imageData = new ImageData
                {
                    ImagePath = strAssetsPath + "wav1" + Path.DirectorySeparatorChar + i + ".png"
                };
                var prediction = predictor.Predict(imageData);
                if (prediction == null || string.IsNullOrEmpty(prediction.PredictedLabelValue))
                {
                    arrNotations[i] = "";
                }
                else
                {
                    if (prediction.PredictedLabelValue == "_GAP_")
                    {
                        prediction.PredictedLabelValue = "";
                    }
                    arrNotations[i] = prediction.PredictedLabelValue;
                }
            });
            textBox4.Text = string.Join(" ", arrNotations);
        }

        private void Tick1()
        {
            if (label2.InvokeRequired)
            {
                label2.Invoke((MethodInvoker)delegate { Tick1(); });
            }
            else
            {
                nCount++;
                label2.Text = nCount.ToString();
            }
        }

        private void button4_Click(object sender, EventArgs e)
        {
            if (textBox4.Text != null)
            {
                Clipboard.SetText(textBox4.Text);
            }
        }

        private void Tick2()
        {
            if (label5.InvokeRequired)
            {
                label5.Invoke((MethodInvoker)delegate { Tick2(); });
            }
            else
            {
                nCount++;
                label5.Text = nCount.ToString();
            }
        }

        private void AppendStatus(string status)
        {
            if (textBox3.InvokeRequired)
            {
                textBox3.Invoke((MethodInvoker)delegate { AppendStatus(status); });
            }
            else
            {
                textBox3.Text = status;
            }
        }
    }

    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }

    public class ImagePrediction : ImageData
    {
        public float[] Score;

        public string PredictedLabelValue;
    }

    struct InceptionSettings
    {
        public const int ImageHeight = 224;
        public const int ImageWidth = 224;
        public const float Mean = 117;
        public const float Scale = 1;
        public const bool ChannelsLast = true;
    }
}
