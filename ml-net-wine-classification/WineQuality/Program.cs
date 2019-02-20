using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace WineQuality
{
    class Program
    {

        //Veri dosyamızdaki her bir satırın C# obje karşılığı olacak sınıfımız
        public class Wine
        {
            [LoadColumn(0)]
            public float FixedAcidity;
            [LoadColumn(1)]
            public float VolatileAcidity;
            [LoadColumn(2)]
            public float CitricAcid;
            [LoadColumn(3)]
            public float ResidualSugar;
            [LoadColumn(4)]
            public float Chlorides;
            [LoadColumn(5)]
            public float FreeSulfurDioxide;
            [LoadColumn(6)]
            public float TotalSulfurDioxide;
            [LoadColumn(7)]
            public float Density;
            [LoadColumn(8)]
            public float PH;
            [LoadColumn(9)]
            public float Sulphates;
            [LoadColumn(10)]
            public float Alcohol;
            [LoadColumn(11)]
            public float Quality;
        }

        // Tahmin iş
        public class WineQualityPrediction
        {
            [KeyType(Count = 3)]
            public uint PredictedLabel;
            public float Label;
            [ColumnName("Score")]
            public float[] Score;
        }

        static void Main(string[] args)
        {
            var context = new MLContext();



            //https://archive.ics.uci.edu/ml/datasets/Wine+Quality
            var dataView = context.Data.ReadFromTextFile<Wine>(path: @"winequality-red.csv",
                                          hasHeader: true,
                                          separatorChar: ';');

            // ÖN İŞLEMLER
            
            var pipeline = context.Transforms.Conversion.ValueMap(new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, // FROM
                                                                  new float[] { 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2 },  // TO
                                                                  (DefaultColumnNames.Label, nameof(Wine.Quality))) // QUALITY den LABEL'a aktar (sağdan sola)

                            // Değerleri -1 ile 1 arasında ifade et
                           .Append(context.Transforms.Normalize(mode: Microsoft.ML.Transforms.Normalizers.NormalizingEstimator.NormalizerMode.MinMax, columns: dataView.Schema.Where(c => c.Name != nameof(Wine.Quality))
                                                                                           .Select(c => (c.Name, c.Name))
                                                                                           .ToArray()))
                           // Öğrenmek için kullanacağım nitelikleri seç (Quality hariç hepsi)
                           .Append(context.Transforms.Concatenate(outputColumnName: "Features",
                                                                  inputColumnNames: dataView.Schema.Where(c => c.Name != nameof(Wine.Quality))
                                                                                           .Select(c => c.Name)
                                                                                           .ToArray()))
                           // Bu noktaya kadar yapılanları cache'le, aynı işlemler tekrar tekrar yapılmasın
                           .AppendCacheCheckpoint(context); ;


            // Veriyi 0.3 oranında ikiye dağıt. (Tuple deconstruction C# 7.1)
            (var trainSet, var testSet) = context.MulticlassClassification.TrainTestSplit(dataView, 0.3, seed: 5);

            // Öğreticiyi belirle
            var trainer = context.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: DefaultColumnNames.Label, featureColumn: DefaultColumnNames.Features);

            // Öğretici ile yeni bir pipeline aç
            var trainingPipeline = pipeline.Append(trainer);

            // Öğren!!!
            var trainedModel = trainingPipeline.Fit(trainSet);

            // Öğrendiğini test verisi üzerine uygula
            var predictions = trainedModel.Transform(testSet);

            // Uygulamanın doğrulu hakkında ölçümler yap
            var metrics = context.MulticlassClassification.Evaluate(predictions, DefaultColumnNames.Label, DefaultColumnNames.Score);

            // BU ölçümleri ekrana bas
            PrintMultiClassClassificationMetrics("", metrics);

            // Öğrendiğin yöntemi Wine alıp WineQualityPrediction döenecek bir Function olarak çevir. 
            var predEngine = trainedModel.CreatePredictionEngine<Wine, WineQualityPrediction>(context);

            // Bu wine nesnesi için bir tahmin yap
            var sample = predEngine.Predict(new Wine
            {
                FixedAcidity = 5.3f,
                VolatileAcidity = 0.47f,
                CitricAcid = 0.11f,
                ResidualSugar = 2.2f,
                Chlorides = 0.048f,
                FreeSulfurDioxide = 16f,
                TotalSulfurDioxide = 89f,
                Density = 0.99f,
                PH = 3.54f,
                Sulphates = 0.88f,
                Alcohol = 13.5f,
                Quality = 0,
            });

            Console.ReadLine();
        }

        public static void PrintMultiClassClassificationMetrics(string name, MultiClassClassifierMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {name} multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.AccuracyMacro:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.AccuracyMicro:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            for (int i = 0; i < metrics.PerClassLogLoss.Length; i++)
            {
                Console.WriteLine($"    LogLoss for class {i} = {metrics.PerClassLogLoss[i]:0.####}, the closer to 0, the better");
            }
            Console.WriteLine($"************************************************************");
        }
    }
}
