import json
from flask import (render_template, url_for, flash,
				   redirect, request, abort, Blueprint)
from flask_login import current_user, login_required
from flaskblog import db
from flaskblog.models import Post
from flaskblog.posts.forms import PredictionForm
from flaskblog.posts.mlmodel.data_preprocess import (convert_entry_df, create_holiday_feature, add_features_datetime_YMD, 
                             				   time_period_bin, own_labelencode, load_run_mlmodel)

posts = Blueprint('posts', __name__)

@posts.route("/prediction/new", methods=['GET', 'POST'])
@login_required
def new_prediction():
	form = PredictionForm()
	if form.validate_on_submit():
		# Preprocessed data submitted and Predict Traffic
		x_pred = convert_entry_df(temp=form.traffic_temp.data, weather_main=form.traffic_weather.data, date_time=form.traffic_datetime.data)
		x_pred = create_holiday_feature(dataset=x_pred, column='date_time', days=3)
		x_pred = add_features_datetime_YMD(x_pred, column="date_time", feature_name=['day', 'time'])
		x_pred = time_period_bin(x_pred, 'time_period')
		dataset_labelencode_weather = json.load(open('./flaskblog/posts/mlmodel/dataset_labelencode_weather.txt'))
		dataset_labelencode_day = json.load(open('./flaskblog/posts/mlmodel/dataset_labelencode_day.txt'))
		dataset_labelencode_time = json.load(open('./flaskblog/posts/mlmodel/dataset_labelencode_time.txt'))
		x_pred = own_labelencode(x_pred, dataset_labelencode_weather, 'weather_main')
		x_pred = own_labelencode(x_pred, dataset_labelencode_day, 'day_of_the_week')
		x_pred = own_labelencode(x_pred, dataset_labelencode_time, 'time_period')
		x_pred = x_pred.iloc[:, :].values
		y_pred = int(load_run_mlmodel('./flaskblog/posts/mlmodel/finalized_model.pkl', x_pred))

		prediction = Post(traffic_datetime=form.traffic_datetime.data, 
						  traffic_temp=form.traffic_temp.data, 
						  traffic_weather=form.traffic_weather.data, 
						  traffic_pred=y_pred, 
						  author=current_user)
		db.session.add(prediction)
		db.session.commit()
		flash('Your prediction is completed!', 'success')
		return redirect(url_for('main.home'))
	return render_template('create_prediction.html', title='New Prediction', form=form, legend='New Prediction')

@posts.route("/prediction/<int:post_id>")
def prediction(post_id):
	post = Post.query.get_or_404(post_id)
	return render_template('prediction.html', title='Prediction', post=post)

#Currently there is some issue with updating of predictions (TO COME BACK AND CHECK)

@posts.route("/prediction/<int:post_id>/update", methods=['GET', 'POST'])
@login_required
def update_prediction(post_id):
	post = Post.query.get_or_404(post_id)
	if post.author != current_user:
		abort(403)
	form = PredictionForm()
	if form.validate_on_submit():
		# Preprocessed data submitted and Predict Traffic
		x_pred = convert_entry_df(temp=form.traffic_temp.data, weather_main=form.traffic_weather.data, date_time=form.traffic_datetime.data)
		x_pred = create_holiday_feature(dataset=x_pred, column='date_time', days=3)
		x_pred = add_features_datetime_YMD(x_pred, column="date_time", feature_name=['day', 'time'])
		x_pred = time_period_bin(x_pred, 'time_period')
		dataset_labelencode_weather = json.load(open('./flaskblog/posts/mlmodel/dataset_labelencode_weather.txt'))
		dataset_labelencode_day = json.load(open('./flaskblog/posts/mlmodel/dataset_labelencode_day.txt'))
		dataset_labelencode_time = json.load(open('./flaskblog/posts/mlmodel/dataset_labelencode_time.txt'))
		x_pred = own_labelencode(x_pred, dataset_labelencode_weather, 'weather_main')
		x_pred = own_labelencode(x_pred, dataset_labelencode_day, 'day_of_the_week')
		x_pred = own_labelencode(x_pred, dataset_labelencode_time, 'time_period')
		x_pred = x_pred.iloc[:, :].values
		y_pred = int(load_run_mlmodel('./flaskblog/posts/mlmodel/finalized_model.pkl', x_pred))

		post.traffic_datetime = form.traffic_datetime.data
		post.traffic_temp = form.traffic_temp.data
		post.traffic_weather = form.traffic_weather.data
		post.traffic_pred = y_pred
		db.session.commit()
		flash('Your prediction has been updated!', 'success')
		return redirect(url_for('posts.prediction', post_id=post.id))
	elif request.method =='GET':
		form.traffic_datetime.data = post.traffic_datetime
		form.traffic_temp.data = post.traffic_temp
		form.traffic_weather.data = post.traffic_weather
	return render_template('create_prediction.html', title='Update Prediction', form=form, legend='Update Prediction')

@posts.route("/prediction/<int:post_id>/delete", methods=['POST'])
@login_required
def delete_prediction(post_id):
	post = Post.query.get_or_404(post_id)
	if post.author != current_user:
		abort(403)
	db.session.delete(post)
	db.session.commit()
	flash('Your prediction has been deleted!', 'success')
	return redirect(url_for('main.home'))

