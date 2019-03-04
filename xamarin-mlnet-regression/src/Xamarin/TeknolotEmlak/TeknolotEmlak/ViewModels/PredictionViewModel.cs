using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Reflection;
using System.Windows.Input;

namespace TeknolotEmlak.ViewModels
{
    public sealed class Home
    {
        public float Price;
        public float Bedrooms;
        public float Bathrooms;
        public float Sqft_Living;
        public float Floors = 1;
        public float Waterfront;
        public float View;
        public float Condition = 1;
        public float Grade = 1;
        public float Yr_Built = 1900;
    }
    public class HousePricePrediction
    {
        [ColumnName("Score")]
        public float Price;
    }

    public class PredictionViewModel : BaseViewModel
    {
        private readonly Microsoft.ML.Core.Data.ITransformer _model;
        private readonly PredictionEngine<Home, HousePricePrediction> _predictionEngine;
        private readonly Home _home;

        public PredictionViewModel()
        {
            _home = new Home();

            Title = "Emlak Tahmin";

            var context = new MLContext();
            var assembly = IntrospectionExtensions.GetTypeInfo(typeof(PredictionViewModel)).Assembly;

            using (var stream = assembly.GetManifestResourceStream("TeknolotEmlak.MLModels.model.zip"))
            {
                _model = context.Model.Load(stream);
                _predictionEngine = _model.CreatePredictionEngine<Home, HousePricePrediction>(context);
            }

            this.PropertyChanged += (s, e) => {
                if(e.PropertyName == nameof(this.Price))
                {
                    return;
                }
                Price = (decimal)_predictionEngine.Predict(_home).Price;
            };
        }

        private decimal _price;

        public decimal Price
        {
            get { return _price; }
            set { SetProperty(ref _price, value); }
        }

        public float Bedrooms
        {
            get => _home.Bedrooms;
            set => SetProperty(ref _home.Bedrooms, (float)Math.Round(value));
        }
        public float Bathrooms
         {
            get => _home.Bathrooms;
            set => SetProperty(ref _home.Bathrooms, (float)Math.Round(value));
         }

        public float Sqft_Living
        {
            get => _home.Sqft_Living;
            set => SetProperty(ref _home.Sqft_Living, (float)Math.Round(value));
        }

        public float Floors
        {
            get => _home.Floors;
            set => SetProperty(ref _home.Floors, (float)Math.Round(value));
        }

        public bool Waterfront
        {
            get => _home.Waterfront == 1.0f;
            set => SetProperty(ref _home.Waterfront, value ? 1f : 0f);
        }

        public float View
        {
            get => _home.View;
            set => SetProperty(ref _home.View, (float)Math.Round(value));
        }

        public float Condition
        {
            get => _home.Condition;
            set => SetProperty(ref _home.Condition, (float)Math.Round(value));
        }

        public float Grade
        {
            get => _home.Grade;
            set => SetProperty(ref _home.Grade, (float)Math.Round(value));
        }
        public float Yr_Built
        {
            get => _home.Yr_Built;
            set => SetProperty(ref _home.Yr_Built, (float)Math.Round(value));
        }
        public ICommand PredictionCommand { get; }
    }
}