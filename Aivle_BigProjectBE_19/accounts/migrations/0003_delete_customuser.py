# Generated by Django 5.0 on 2023-12-18 08:47

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0002_remove_customuser_user_id_alter_customuser_username'),
    ]

    operations = [
        migrations.DeleteModel(
            name='CustomUser',
        ),
    ]
