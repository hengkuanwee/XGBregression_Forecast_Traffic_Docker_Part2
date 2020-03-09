from datetime import datetime
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from flask import Flask, current_app
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = "5f06cab0a14e1d2e39bf7b9cacdc98f5"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

class User(db.Model, UserMixin):
	id = db.Column(db.Integer, primary_key=True)
	username = db.Column(db.String(20), unique=True, nullable=False)
	email = db.Column(db.String(120), unique=True, nullable=False)
	image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
	password = db.Column(db.String(60), nullable=False)
	posts = db.relationship('Post', backref='author', lazy=True)

	def get_reset_token(self, expires_sec=1800):
		s = Serializer(current_app.config['SECRET_KEY'], expires_sec)
		return s.dumps({'user_id': self.id}).decode('utf-8')

	@staticmethod
	def verify_reset_token(token):
		s = Serializer(current_app.config['SECRET_KEY'])
		try:
			user_id = s.loads(token)['user_id']
		except:
			return None
		return User.query.get(user_id)

	def __repr__(self):
		return f"User('{self.username}', '{self.email}', '{self.image_file}')"

class Post(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	traffic_datetime = db.Column(db.DateTime, nullable=False)
	traffic_temp = db.Column(db.Integer, nullable=False)
	traffic_weather = db.Column(db.Text, nullable=False)
	traffic_pred = db.Column(db.Integer, nullable=False)
	date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
	user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
	def __repr__(self):
		return f"Predicted traffic on Post('{self.date_time}', '{self.traffic_pred}')"
