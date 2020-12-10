from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()

def init_db(app):
    db.init_app(app)


def create_db(app):
    with app.app_context():
        db.create_all()


def insert_hourly_data(app, date, time, total_people, male, female, age1_16, age17_30, age31_45, age46_60):
    tracking = TrackByHour(date, time, total_people, male, female, age1_16, age17_30, age31_45, age46_60)
    with app.app_context():
        db.session.add(tracking)
        db.session.commit()


def get_daily_data(app, date):
    tracking_info = None
    with app.app_context():
        tracking_info = TrackByDay.query.filter_by(date=date).first()
    return tracking_info


def get_hourly_data(app, date, time):
    tracking_info = None
    with app.app_context():
        tracking_info = TrackByHour.query.filter_by(date=date, time=time).first()
    return tracking_info


def get_hourly_stats(app, date, time):
    tracking_info = None
    with app.app_context():
        tracking_info = TrackByHour.query.filter_by(date=date, time=time).all()
    return tracking_info


def delete_row(app, total_people):
    with app.app_context():
        tracking_info = TrackByDay.query.filter_by(total_people=total_people).first()
        db.session.delete(tracking_info)
        db.session.commit()


def get_daily_stats(app, date):
    tracking_info = None
    with app.app_context():
        tracking_info = TrackByDay.query.filter_by(date=date).all()
    return tracking_info


def update_daily_data(app, date, total_people, male, female, age1_16, age17_30, age31_45, age46_60):
    with app.app_context():
        tracking_info = TrackByDay.query.filter_by(date=date).first()
        if not tracking_info:
            insert_daily_data(app, date, total_people, male, female, age1_16, age17_30, age31_45, age46_60)
        else:
            tracking_info.total_people = total_people
            tracking_info.male = male
            tracking_info.female = female
            tracking_info.age1_16 = age1_16
            tracking_info.age17_30 = age17_30
            tracking_info.age31_45 = age31_45
            tracking_info.age46_60 = age46_60
            db.session.commit()


def update_hourly_data(app, date, time, total_people, male, female, age1_16, age17_30, age31_45, age46_60):
    with app.app_context():
        tracking_info = TrackByHour.query.filter_by(time=time, date=date).first()
        if not tracking_info:
            insert_hourly_data(app, date, time, total_people, male, female, age1_16, age17_30, age31_45, age46_60)
        else:
            tracking_info.total_people = total_people
            tracking_info.male = male
            tracking_info.female = female
            tracking_info.age1_16 = age1_16
            tracking_info.age17_30 = age17_30
            tracking_info.age31_45 = age31_45
            tracking_info.age46_60 = age46_60
            db.session.commit()


def insert_daily_data(app, date, total_people, male, female, age1_16, age17_30, age31_45, age46_60):
    tracking = TrackByDay(date, total_people, male, female, age1_16, age17_30, age31_45, age46_60)
    with app.app_context():
        db.session.add(tracking)
        db.session.commit()


class TrackByHour(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    total_people = db.Column(db.Integer, unique=False, nullable=False)
    age1_16 = db.Column(db.Integer, unique=False, nullable=False)
    age17_30 = db.Column(db.Integer, unique=False, nullable=False)
    age31_45 = db.Column(db.Integer, unique=False, nullable=False)
    age46_60 = db.Column(db.Integer, unique=False, nullable=False)
    female = db.Column(db.Integer, unique=False, nullable=False)
    male = db.Column(db.Integer, unique=False, nullable=False)

    def __repr__(self):
        return '<Hour>'

    def __init__(self, date, time, total_people, male, female, age1_16, age17_30, age31_45, age46_60):
        self.total_people = total_people
        self.date = date
        self.time = time
        self.age1_16 = age1_16
        self.age17_30 = age17_30
        self.age31_45 = age31_45
        self.age46_60 = age46_60
        self.female = female
        self.male = male


class TrackByDay(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    total_people = db.Column(db.Integer, unique=False, nullable=False)
    age1_16 = db.Column(db.Integer, unique=False, nullable=False)
    age17_30 = db.Column(db.Integer, unique=False, nullable=False)
    age31_45 = db.Column(db.Integer, unique=False, nullable=False)
    age46_60 = db.Column(db.Integer, unique=False, nullable=False)
    female = db.Column(db.Integer, unique=False, nullable=False)
    male = db.Column(db.Integer, unique=False, nullable=False)

    def __repr__(self):
        return '<Day>'

    def __init__(self, date, total_people, male, female, age1_16, age17_30, age31_45, age46_60):
        self.date = date
        self.total_people = total_people
        self.age1_16 = age1_16
        self.age17_30 = age17_30
        self.age31_45 = age31_45
        self.age46_60 = age46_60
        self.female = female
        self.male = male
