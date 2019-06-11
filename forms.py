from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, SelectField, IntegerField
from wtforms.validators import DataRequired, Length, Email, EqualTo

from os import listdir

class GenerationSettings(FlaskForm):
	texts = listdir("./texts/")
	text_choices = []
	for text in texts:
		text = text.split('.')[0]
		text_choices.append((text,text))

	models = listdir("./exports/models/")
	model_choices = []
	for model in models:
		if "h5" in model and "generator" in model:
			model = model.split('generator_')[1].split('.')[0]
			model_choices.append((model,model))

	w2ms = listdir("./exports/word2vec/")
	w2m_choices = []
	for w2m in w2ms:
		w2m_choices.append((w2m,w2m))


	model_to_load = SelectField(u'Trained model', choices=model_choices)
	word2vec_model = SelectField(u'Word2Vector model', choices=w2m_choices)
	text_to_imitate = SelectField(u'Text to imitate', choices=text_choices)
	number_of_lines = IntegerField(u'Number of lines to generate')
	submit = SubmitField('Generate')