# Generated by Django 5.0 on 2024-01-02 06:35

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0010_remove_profile_rank'),
        ('board', '0005_alter_comment_comment'),
    ]

    operations = [
        migrations.AddField(
            model_name='comment',
            name='profile',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='accounts.profile'),
        ),
    ]
