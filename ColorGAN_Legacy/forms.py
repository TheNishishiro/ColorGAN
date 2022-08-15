from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, SelectField, IntegerField
from wtforms.validators import DataRequired, Length, Email, EqualTo

from os import listdir

class GenerationSettings(FlaskForm):


	models = listdir("./models/")
	model_choices = []
	for model in models:
		if "h5" in model and "g_" in model:
			model = model.split('g_')[1].split('.')[0]
			model_choices.append((model,model))


	model_to_load = SelectField(u'Trained model', choices=model_choices)