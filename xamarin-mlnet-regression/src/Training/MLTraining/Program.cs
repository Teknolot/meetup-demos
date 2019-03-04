using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace MLTraining
{
    public sealed class Home
    {
        [LoadColumn(2)]
        public float Price;
        [LoadColumn(3)]
        public float Bedrooms;
        [LoadColumn(4)]
        public float Bathrooms;
        [LoadColumn(5)]
        public float Sqft_Living;
        [LoadColumn(7)]
        public float Floors;
        [LoadColumn(8)]
        public float Waterfront;
        [LoadColumn(9)]
        public float View;
        [LoadColumn(10)]
        public float Condition;
        [LoadColumn(11)]
        public float Grade;
        [LoadColumn(14)]
        public float Yr_Built;
    }
    public class HousePricePrediction
    {
        [ColumnName("Score")]
        public float Price;
    }

    public class Program
    {
        public static void Main(string[] args)
        {

            // ML Contexti oluştur
            var context = new MLContext();
            // https://www.kaggle.com/harlfoxem/housesalesprediction
            // Veriyi oku
            var data = context.Data.ReadFromTextFile<Home>(path: "house_data.csv",
                                                           separatorChar: ',',
                                                           hasHeader: true,
                                                           trimWhitespace: true);
            // Veriyi eğitim ve test olarak ikiye ayır
            (var trainSet, var testSet) = context.Regression.TrainTestSplit(data, 0.3, seed: 5);

            //İş hattını oluştur
            //Price kolonunun bir kopyasını Label olarak oluştur
            var learningPipeline = context.Transforms.CopyColumns(inputColumnName: nameof(Home.Price),
                                                                  outputColumnName: DefaultColumnNames.Label)
                                    // Price ve Label hariç tüm kolonlar niteliktir
                                    .Append(context.Transforms.Concatenate(inputColumnNames: data.Schema
                                                                                                      .Select(c => c.Name)
                                                                                                      .Except(new[] { nameof(Home.Price) ,
                                                                                                                      DefaultColumnNames.Label})
                                                                                                      .ToArray(),
                                                                          outputColumnName: DefaultColumnNames.Features
                                                                          )
                                          )
                                    //Öğrenme algoritması seçimi
                                   .Append(context.Regression.Trainers.FastTree(labelColumn: DefaultColumnNames.Label,
                                                                                featureColumn: DefaultColumnNames.Features))
                                   .AppendCacheCheckpoint(context);
            // Öğren
            var model = learningPipeline.Fit(trainSet);
            // Öğrendiğini test verisine uygula
            var predictions = model.Transform(testSet);
            
            // Öğrenmenin kalitesini ölç
            var metrics = context.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine($"RSquared = {metrics.RSquared}");

            // Modeli kaydet
            using (var stream = System.IO.File.OpenWrite("model.zip"))
            {
                model.SaveTo(context, stream);
            }
        }
    }
}
