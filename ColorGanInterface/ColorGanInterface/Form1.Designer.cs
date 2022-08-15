namespace ColorGanInterface
{
	partial class Form1
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}

			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.button1 = new System.Windows.Forms.Button();
			this.pictureBoxGray = new System.Windows.Forms.PictureBox();
			this.pictureBoxColor = new System.Windows.Forms.PictureBox();
			((System.ComponentModel.ISupportInitialize) (this.pictureBoxGray)).BeginInit();
			((System.ComponentModel.ISupportInitialize) (this.pictureBoxColor)).BeginInit();
			this.SuspendLayout();
			// 
			// button1
			// 
			this.button1.Location = new System.Drawing.Point(12, 12);
			this.button1.Name = "button1";
			this.button1.Size = new System.Drawing.Size(518, 23);
			this.button1.TabIndex = 0;
			this.button1.Text = "Upload";
			this.button1.UseVisualStyleBackColor = true;
			this.button1.Click += new System.EventHandler(this.button1_Click);
			// 
			// pictureBoxGray
			// 
			this.pictureBoxGray.Location = new System.Drawing.Point(12, 41);
			this.pictureBoxGray.Name = "pictureBoxGray";
			this.pictureBoxGray.Size = new System.Drawing.Size(256, 256);
			this.pictureBoxGray.TabIndex = 1;
			this.pictureBoxGray.TabStop = false;
			// 
			// pictureBoxColor
			// 
			this.pictureBoxColor.Location = new System.Drawing.Point(274, 41);
			this.pictureBoxColor.Name = "pictureBoxColor";
			this.pictureBoxColor.Size = new System.Drawing.Size(256, 256);
			this.pictureBoxColor.TabIndex = 2;
			this.pictureBoxColor.TabStop = false;
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(544, 313);
			this.Controls.Add(this.pictureBoxColor);
			this.Controls.Add(this.pictureBoxGray);
			this.Controls.Add(this.button1);
			this.Name = "Form1";
			this.Text = "Form1";
			((System.ComponentModel.ISupportInitialize) (this.pictureBoxGray)).EndInit();
			((System.ComponentModel.ISupportInitialize) (this.pictureBoxColor)).EndInit();
			this.ResumeLayout(false);
		}

		private System.Windows.Forms.Button button1;
		private System.Windows.Forms.PictureBox pictureBoxGray;
		private System.Windows.Forms.PictureBox pictureBoxColor;

		#endregion
	}
}