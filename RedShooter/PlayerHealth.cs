using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerHealth : MonoBehaviour
{
    public int currentHealth;
    public int maxHealth;
    public float immortalTime;
    private float immortalCounter;

    private Vector2 checkPoint;

    public static PlayerHealth instance;
    public HealthBar healthbar;

    private SpriteRenderer sr;

    void Awake()
    {
        instance = this;
    }

    void Start()
    {
        checkPoint = transform.position;

        currentHealth = maxHealth;
        healthbar.SetMaxHealth(currentHealth);

        sr = GetComponent<SpriteRenderer>();
    }

    void Update()
    {
        if(immortalCounter > 0)
        {
            immortalCounter -= Time.deltaTime;

            if(immortalCounter <= 0)
            {
                sr.color = new Color(sr.color.r, sr.color.g, sr.color.b, 1f); 
            }
        }
    }

    public void DealDamage()
    {
        if(immortalCounter <= 0) 
        {
            currentHealth--;

            healthbar.SetHealth(currentHealth);

            if (currentHealth <= 0)
            {
                Die();
                // gameObject.SetActive(false);
            }
            else
            {
                immortalCounter = immortalTime;
                sr.color = new Color(sr.color.r, sr.color.g, sr.color.b, 0.6f); 
            }
        }
    }

    public void UpdatedCheckpoint(Vector2 pos)
    {
        checkPoint = pos;
    }

    void Die()
    {
        Respawn();
    }

    void Respawn()
    {
        currentHealth = maxHealth;
        healthbar.SetHealth(currentHealth);

        transform.position = checkPoint;
    }
}
