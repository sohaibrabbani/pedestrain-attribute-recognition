from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()


def init_db(app):
    db.init_app(app)


def create_db(app):
    with app.app_context():
        db.create_all()


def insert_hourly_data(app, date, time, total_people, male, female, age1_16, age_17_30, age31_45, age46_60):
    tracking = TrackByHour(date, time, total_people, male, female, age1_16, age_17_30, age31_45, age46_60)
    with app.app_context():
        db.session.add(tracking)
        db.session.commit()


class TrackByHour(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    total_people = db.Column(db.Integer, unique=False, nullable=False)
    age1_16 = db.Column(db.Integer, unique=False, nullable=False)
    age_17_30 = db.Column(db.Integer, unique=False, nullable=False)
    age31_45 = db.Column(db.Integer, unique=False, nullable=False)
    age46_60 = db.Column(db.Integer, unique=False, nullable=False)
    female = db.Column(db.Integer, unique=False, nullable=False)
    male = db.Column(db.Integer, unique=False, nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username

    def __init__(self, date, time, total_people, male, female, age1_16, age_17_30, age31_45, age46_60):
        self.total_people = total_people
        self.date = date
        self.time = time
        self.age1_16 = age1_16
        self.age_17_30 = age_17_30
        self.age31_45 = age31_45
        self.age46_60 = age46_60
        self.female = female
        self.male = male
