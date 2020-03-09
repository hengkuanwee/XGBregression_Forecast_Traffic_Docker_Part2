from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, DateTimeField, IntegerField, SelectField
from wtforms.validators import InputRequired, ValidationError
import datetime

class PredictionForm(FlaskForm):
	traffic_datetime = DateTimeField('DateTime (e.g. 2020-12-31 17:00:00)', format='%Y-%m-%d %H:%M:%S', validators=[InputRequired()])
	traffic_temp = IntegerField('Temperature (in ' + u'\N{DEGREE SIGN}' + 'F)', validators=[InputRequired()])
	traffic_weather = SelectField('Weather', choices=[('Clear','Clear'), ('Clouds','Clouds'), ('Drizzle','Drizzle'), 
								  ('Fog','Fog'), ('Haze','Haze'), ('Mist','Mist'), ('Rain','Rain'), ('Snow','Snow'), 
								  ('Squall','Squall'), ('Thunderstorm','Thunderstorm')], 
								  validators=[InputRequired()])
	submit = SubmitField('Post')

	def validate_traffic_datetime(self, traffic_datetime):
		if isinstance(traffic_datetime.data, datetime.datetime):
			if traffic_datetime.data < datetime.datetime(year=1900, month=1, day=1, hour=0, minute=0, second=0):
				raise ValidationError('Please enter a date after 1900s')
			if traffic_datetime.data >= datetime.datetime(year=2090, month=1, day=1, hour=0, minute=0, second=0):
				raise ValidationError('Please enter a date before 2090s')
		else:
			raise ValidationError('')